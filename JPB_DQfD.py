import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Dense, Activation, Flatten, Input, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from collections import deque
from helpers.action import Action
import pickle
import pandas as pd

class DQfD():
    # Exploration Decay determines how quickly the agent stops exploring (closer to 0 means much quicker while closer to 1 means much slower)
    # Exploriation Min is the abolsute minimum exploriation rate that the agent can reach so that it doesn't completely stop exploration at any point
    # Exploration Rate determines how often the agent decides to explore and take alternate options
    # Discount Rate determines how valuable the agent thinks each reward is (closer to 0 means the current reward is more important while closer to 1 means that a future reward is more important)
    # NSteps represent how far in the future rewards will be considered for nstep loss
    # Margin determines the margin of error to add to non-expert actions when training expert experiences
    def __init__(self, observationSpace, actionSpace, learnRate, discRate, explorationMin, explorationMax, explorationDecay, nsteps, margin, batchSize, memSize, verbose):
        self.explorationDecay = explorationDecay
        self.explorationMin = explorationMin
        self.explorationRate = explorationMax
        self.learnRate = learnRate
        self.discRate = discRate
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        self.batchSize = batchSize
        self.expertBatchSize = int(self.batchSize / 10)
        self.selfGeneratedBatchSize = self.batchSize - self.expertBatchSize
        self.nsteps = nsteps
        self.margin = margin
        self.memSize = memSize
        self.expertBuffer = deque(maxlen=1000)
        self.memoryBuffer = deque(maxlen=self.memSize)
        self.verbose = verbose
        self.lambdaConst = [1.0, 1.0, 1.0]

    def tdLoss(self, yTrue, yPred):
        sqLoss = K.square(yPred - yTrue)
        loss = K.mean(sqLoss, axis=-1)
        return loss

    def supervisedLoss(self, yTrue, yPred):
        absLoss = K.abs(yPred - yTrue)
        loss = K.mean(absLoss, axis=-1)
        return loss

    # Creates the models that the agent will use in different steps of the DQfD process (expert training, double DQN and DQfD)
    def createModel(self, learnRate, l2Factor=0.001, hidden1=16, hidden2=32):
        inputLayer = Input(shape=(self.observationSpace,))
        dense1 = Dense(hidden1, activation="relu", kernel_regularizer=regularizers.l2(l2Factor))(inputLayer)
        dense2 = Dense(hidden2, activation="relu", kernel_regularizer=regularizers.l2(l2Factor))(dense1)
        dense3 = Dense(hidden1, activation="relu", kernel_regularizer=regularizers.l2(l2Factor))(dense2)
        outputLayer = Dense(self.actionSpace, activation="linear", kernel_regularizer=regularizers.l2(l2Factor))(dense3)
        tempModel = Model(inputLayer, outputLayer)
        inputDQ = Input(shape=(self.observationSpace,), dtype='float32')
        inputNstep = Input(shape=(self.observationSpace,), dtype='float32')
        inputSLMC = Input(shape=(self.observationSpace,), dtype='float32')
        outputDQ = tempModel(inputDQ)
        outputNStep = tempModel(inputNstep)
        outputSLMC = tempModel(inputSLMC)
        model = Model(inputs=[inputDQ, inputNstep, inputSLMC], outputs=[outputDQ, outputNStep, outputSLMC])
        model.compile(loss=[self.tdLoss, self.tdLoss, self.supervisedLoss], loss_weights=self.lambdaConst, optimizer=Adam(lr=learnRate))
        return model

    # Initial models for pretraining
    def createDQNModels(self):
        self.evalModel = self.createModel(self.learnRate)
        self.targetModel = self.createModel(self.learnRate)

    # Store experiences that can be sampled from at a later point
    def storeExperience(self, bufferToAppend, currentState, action, reward, nextObs, done, isExpert):
        nstep = 0
        nstepObs = currentState
        nstepReward = 0
        bufferToAppend.append([currentState, action, reward, nextObs, done, isExpert, nstep, nstepObs, nstepReward])
        bufferLength = len(bufferToAppend)

        if(bufferLength > self.nsteps):
            indexToUpdate = bufferLength-self.nsteps
            for memIndex in range (indexToUpdate, bufferLength):
                if(bufferToAppend[memIndex][4]):
                    break
                nstep += 1
                nstepObs = bufferToAppend[memIndex][0]
                nstepReward += ((self.discRate**nstep) * bufferToAppend[memIndex][2])
            bufferToAppend[indexToUpdate][6] = nstep
            bufferToAppend[indexToUpdate][7] = nstepObs
            bufferToAppend[indexToUpdate][8] = nstepReward
        
            if(bufferToAppend == self.memoryBuffer):
                self.singleReplay(self.memoryBuffer[indexToUpdate])

    # Loading expert data from a csv file into a memory buffer
    def loadExpertData(self):
        expertData = pd.read_csv('expert.csv')
        for index, row in expertData.iterrows():
            currentState = [row['current dealer'], row['current ace'], row['current total'], row['current split']]
            nextObs = [row['next dealer'], row['next ace'], row['next total'], row['next split']]
            currentState = np.reshape(currentState, [1, self.observationSpace])
            nextObs = np.reshape(nextObs, [1, self.observationSpace])
            self.storeExperience(self.expertBuffer, currentState, row['action'], row['reward'], nextObs, row['done'], True)  

    # Update target model weights based on evaluation model
    def updateTargetWeights(self, isPreTrain):
        if(isPreTrain):
            weights = self.evalModel.get_weights()
            self.targetModel.set_weights(weights)   
        else:
            weights = self.altModel.get_weights()
            self.trainableModel.set_weights(weights)

    # Save the pretraining model and load it into a new useable model    
    def loadTrainedModel(self, fullTrained=False, fullTrainedName=None):
        if(fullTrained):
            self.trainableModel = self.createModel(self.learnRate)
            self.trainableModel.load_weights(fullTrainedName)
        else:
            self.targetModel.save_weights("pretrainedModel.h5")
            self.trainableModel = self.createModel(self.learnRate)
            self.altModel = self.createModel(self.learnRate)
            self.trainableModel.load_weights("pretrainedModel.h5")
            self.altModel.load_weights("pretrainedModel.h5")

    # Get batch for training
    def getSamples(self):
        batch = []
        batchIndex = []

        # Pull random experiences of batchSize to sample from and update
        while len(batch) < (self.batchSize - self.expertBatchSize):
            index = random.randrange(0, len(self.memoryBuffer))
            indexToSample = [index, False]
            if indexToSample not in batchIndex:
                batch.append(self.memoryBuffer[index])
                batchIndex.append(indexToSample)

        while len(batch) < self.batchSize:
            index = random.randrange(0, len(self.expertBuffer))
            indexToSample = [index, True]
            if indexToSample not in batchIndex:
                batch.append(self.expertBuffer[index])
                batchIndex.append(indexToSample)

        return batch, batchIndex

    # Decide an action to perform
    # Either random exploration or the best predicted result
    def decideAction(self, currentState):
        if np.random.rand() < self.explorationRate:
            if(currentState[0][3]):
                return random.randrange(self.actionSpace)
            else:
                return random.randrange(self.actionSpace - 1)
        dqPred, nstepPred, slmcPred = self.trainableModel.predict([currentState, currentState, currentState])
        QValues = dqPred + nstepPred + slmcPred
        chosenAction = np.argmax(QValues[0])
        # If the split action is chosen, but the max number of hands are already in play then change the action to the next best option
        if(chosenAction == Action.SPLIT.value and currentState[0][3] == False):            
            newQValues = [QValues[0][0], QValues[0][2], QValues[0][2]]
            chosenAction = np.argmax(newQValues)
        return chosenAction

    # Updates model with a single replay. Assures that each replay is trained on atleast once
    def singleReplay(self, replay):
        currentState, action, reward, nextObs, done, isExpert, nstep, nstepObs, nstepReward = replay
        QValsNextTarget, nstepQValsNextTarget, slmcQValsCurrentTarget = self.trainableModel.predict([nextObs, nstepObs, currentState])
        QValsNextEval, nstepQValsNextEval, slmcQValsCurrentEval = self.altModel.predict([nextObs, nstepObs, currentState])
        bestAction = np.argmax(QValsNextEval, axis=1)
        nstepBestAction = np.argmax(nstepQValsNextEval, axis=1)
        expertAction = np.argmax(slmcQValsCurrentEval, axis=1)
        expertActionQVal = np.amax(slmcQValsCurrentEval, axis=1)
        actualDQVals, actualNStepQVals, actualSLMCQVals = self.altModel.predict([currentState, currentState, currentState])
        actualDQVals[0, action] = reward + ((abs(1 - done)) * self.discRate * QValsNextTarget[0, bestAction])
        actualNStepQVals[0, action] = nstepReward + ((abs(1 - done)) * (self.discRate ** nstep) * nstepQValsNextTarget[0, nstepBestAction])

        if(bestAction == Action.SPLIT.value and nextObs[0, 3] == False):
            actualDQVals[0, Action.SPLIT.value] = (np.amin(actualDQVals) -1)
    
        if(nstepBestAction == Action.SPLIT.value and nstepObs[0, 3] == False):
            actualNStepQVals[0, Action.SPLIT.value] = (np.amin(actualNStepQVals) -1)

        if(expertAction == Action.SPLIT.value and currentState[0][3] == False):
            actualSLMCQVals[0, expertAction] = (np.amin(actualSLMCQVals) -1)
        
        self.altModel.fit([currentState, currentState, currentState], [actualDQVals, actualNStepQVals, actualSLMCQVals], verbose=self.verbose)
        self.updateTargetWeights(False)

    # Pre-Training the network using expert data (train on batch)
    def preTraining(self):
        index = list(range(len(self.expertBuffer)))
        currentStates = np.asarray([self.expertBuffer[i][0][0] for i in range(0, len(self.expertBuffer))])
        actions = np.asarray([self.expertBuffer[i][1] for i in range(0, len(self.expertBuffer))])
        rewards = np.asarray([self.expertBuffer[i][2] for i in range(0, len(self.expertBuffer))])
        nextObs = np.asarray([self.expertBuffer[i][3][0] for i in range(0, len(self.expertBuffer))])
        dones = np.asarray([self.expertBuffer[i][4] for i in range(0, len(self.expertBuffer))])
        isExperts = np.asarray([self.expertBuffer[i][5] for i in range(0, len(self.expertBuffer))])
        nsteps = np.asarray([self.expertBuffer[i][6] for i in range(0, len(self.expertBuffer))])
        nstepObs = np.asarray([self.expertBuffer[i][7][0] for i in range(0, len(self.expertBuffer))])
        nstepRewards = np.asarray([self.expertBuffer[i][8] for i in range(0, len(self.expertBuffer))])

        QValsNextTarget, nstepQValsNextTarget, slmcQValsCurrentTarget = self.targetModel.predict([nextObs, nstepObs, currentStates])
        QValsNextEval, nstepQValsNextEval, slmcQValsCurrentEval = self.evalModel.predict([nextObs, nstepObs, currentStates])
        bestAction = np.argmax(QValsNextEval, axis=1)
        nstepBestAction = np.argmax(nstepQValsNextEval, axis=1)
        expertActionQVal = np.amax(slmcQValsCurrentEval, axis=1)
        actualDQVals, actualNStepQVals, actualSLMCQVals = self.evalModel.predict([currentStates, currentStates, currentStates])
        actualDQVals[index, actions] = rewards + ((1 - dones) * self.discRate * QValsNextTarget[np.arange(len(self.expertBuffer)), bestAction])
        actualNStepQVals[index, actions] = nstepRewards + ((1 - dones) * (self.discRate ** nsteps) * nstepQValsNextTarget[np.arange(len(self.expertBuffer)), nstepBestAction])
        
        count = 0
        for i in expertActionQVal:
            tempRow = np.zeros((1, self.actionSpace))
            tempRow = np.full_like(tempRow, i) + self.margin
            tempRow[0, actions[count]] = i
            actualSLMCQVals[count] = tempRow
            count += 1

        for index in range(len(actualDQVals)):
            tempState = currentStates[index]
            tempState = np.expand_dims(tempState, axis=0)
            tempDQ = actualDQVals[index]
            tempDQ = np.expand_dims(tempDQ, axis=0)
            tempNstep = actualNStepQVals[index]
            tempNstep = np.expand_dims(tempNstep, axis=0)
            tempSLMC = actualSLMCQVals[index]
            tempSLMC = np.expand_dims(tempSLMC, axis=0)            
            self.evalModel.fit([tempState, tempState, tempState], [tempDQ, tempNstep, tempSLMC], verbose=self.verbose)
        #self.evalModel.fit([currentStates, currentStates, currentStates], [actualDQVals, actualNStepQVals, actualSLMCQVals])
        self.updateTargetWeights(True)

    # Normal experience replay, updates multiple experiences at once (faster, less accuracte)
    def experienceReplay(self):
        # Memory size is too small to pull from
        if len(self.memoryBuffer) < self.batchSize:
            return

        batch, batchIndex = self.getSamples()
        index = list(range(self.batchSize))
        currentStates = np.asarray([batch[i][0][0] for i in range(0, self.batchSize)])
        actions = np.asarray([batch[i][1] for i in range(0, self.batchSize)])
        rewards = np.asarray([batch[i][2] for i in range(0, self.batchSize)])
        nextObs = np.asarray([batch[i][3][0] for i in range(0, self.batchSize)])
        dones = np.asarray([batch[i][4] for i in range(0, self.batchSize)])
        isExperts = np.asarray([batch[i][5] for i in range(0, self.batchSize)])
        nsteps = np.asarray([batch[i][6] for i in range(0, self.batchSize)])
        nstepObs = np.asarray([batch[i][7][0] for i in range(0, self.batchSize)])
        nstepRewards = np.asarray([batch[i][8] for i in range(0, self.batchSize)])

        QValsNextTarget, nstepQValsNextTarget, slmcQValsCurrentTarget = self.trainableModel.predict([nextObs, nstepObs, currentStates])
        QValsNextEval, nstepQValsNextEval, slmcQValsCurrentEval = self.altModel.predict([nextObs, nstepObs, currentStates])
        bestAction = np.argmax(QValsNextEval, axis=1)
        nstepBestAction = np.argmax(nstepQValsNextEval, axis=1)
        expertAction = np.argmax(slmcQValsCurrentEval, axis=1)
        expertActionQVal = np.amax(slmcQValsCurrentEval, axis=1)
        actualDQVals, actualNStepQVals, actualSLMCQVals = self.altModel.predict([currentStates, currentStates, currentStates])
        actualDQVals[index, actions] = rewards + ((1 - dones) * self.discRate * QValsNextTarget[np.arange(self.batchSize), bestAction])
        actualNStepQVals[index, actions] = nstepRewards + ((1 - dones) * (self.discRate ** nsteps) * nstepQValsNextTarget[np.arange(self.batchSize), nstepBestAction])
        
        for x in range(len(bestAction)):
            if(bestAction[x] == Action.SPLIT.value and nextObs[x][3] == False):
                actualDQVals[x, Action.SPLIT.value] = (np.amin(actualDQVals[x]) -1)
        
            if(nstepBestAction[x] == Action.SPLIT.value and nstepObs[x][3] == False):
                actualNStepQVals[x, 3] = (np.amin(actualNStepQVals[x]) -1)

        count = 0
        for i in expertActionQVal:
            if(batchIndex[count][1] == True):
                tempRow = np.zeros((1, self.actionSpace))
                tempRow = np.full_like(tempRow, i) + self.margin
                tempRow[0, actions[count]] = i
                actualSLMCQVals[count] = tempRow
            else:
                if(expertAction[count] == Action.SPLIT.value and currentStates[count][3] == False):                
                    tempRow = actualSLMCQVals[count]
                    tempRow[expertAction[count]] = (np.amin(tempRow) -1)
                    actualSLMCQVals[count] = tempRow
            
            count += 1
        self.altModel.fit([currentStates, currentStates, currentStates], [actualDQVals, actualNStepQVals, actualSLMCQVals], verbose=self.verbose)
        self.updateTargetWeights(False)    
        self.explorationRate *= self.explorationDecay
        if(self.explorationRate < self.explorationMin):
            self.explorationRate = self.explorationMin

    # Pre-Training the network using expert data (train on each)
    """def preTraining(self):
        count = 0
        for currentState, action, reward, nextObs, done, isExpert, nstep, nstepObs, nstepReward in self.expertBuffer:

            QValsNextTarget, nstepQValsNextTarget, slmcQValsCurrentTarget = self.targetModel.predict([nextObs, nstepObs, currentState])
            QValsNextEval, nstepQValsNextEval, slmcQValsCurrentEval = self.evalModel.predict([nextObs, nstepObs, currentState])
            bestAction = np.argmax(QValsNextEval, axis=1)
            nstepBestAction = np.argmax(nstepQValsNextEval, axis=1)
            expertAction = np.argmax(slmcQValsCurrentEval, axis=1)
            expertActionQVal = np.amax(slmcQValsCurrentEval, axis=1)
            actualDQVals, actualNStepQVals, actualSLMCQVals = self.evalModel.predict([currentState, currentState, currentState])
            actualDQVals[0, action] = reward + ((abs(1 - done)) * self.discRate * QValsNextTarget[0, bestAction])
            actualNStepQVals[0, action] = nstepReward + ((abs(1 - done)) * (self.discRate ** nstep) * nstepQValsNextTarget[0, nstepBestAction])

            if(bestAction == Action.SPLIT.value and nextObs[0, 3] == False):
                actualDQVals[0, Action.SPLIT.value] = (np.amin(actualDQVals) -1)
        
            if(nstepBestAction == Action.SPLIT.value and nstepObs[0, 3] == False):
                actualNStepQVals[0, Action.SPLIT.value] = (np.amin(actualNStepQVals) -1)

            actualSLMCQVals = np.full_like(actualSLMCQVals, expertActionQVal) + self.margin
            actualSLMCQVals[0, action] = expertActionQVal
            
            self.evalModel.fit([currentState, currentState, currentState], [actualDQVals, actualNStepQVals, actualSLMCQVals])
            count += 1
        self.updateTargetWeights(True)



    # Normal experience replay, updates one experiences at a time (slower, less accurate)
    def experienceReplay(self):
        # Memory size is too small to pull from
        if len(self.memoryBuffer) < self.batchSize:
            return

        count = 0
        batch, batchIndex = self.getSamples()

        for currentState, action, reward, nextObs, done, isExpert, nstep, nstepObs, nstepReward in batch:

            QValsNextTarget, nstepQValsNextTarget, slmcQValsCurrentTarget = self.trainableModel.predict([nextObs, nstepObs, currentState])
            QValsNextEval, nstepQValsNextEval, slmcQValsCurrentEval = self.altModel.predict([nextObs, nstepObs, currentState])
            bestAction = np.argmax(QValsNextEval, axis=1)
            nstepBestAction = np.argmax(nstepQValsNextEval, axis=1)
            expertAction = np.argmax(slmcQValsCurrentEval, axis=1)
            expertActionQVal = np.amax(slmcQValsCurrentEval, axis=1)
            actualDQVals, actualNStepQVals, actualSLMCQVals = self.altModel.predict([currentState, currentState, currentState])
            actualDQVals[0, action] = reward + ((abs(1 - done)) * self.discRate * QValsNextTarget[0, bestAction])
            actualNStepQVals[0, action] = nstepReward + ((abs(1 - done)) * (self.discRate ** nstep) * nstepQValsNextTarget[0, nstepBestAction])

            if(bestAction == Action.SPLIT.value and nextObs[0, 3] == False):
                actualDQVals[0, Action.SPLIT.value] = (np.amin(actualDQVals) -1)
        
            if(nstepBestAction == Action.SPLIT.value and nstepObs[0, 3] == False):
                actualNStepQVals[0, Action.SPLIT.value] = (np.amin(actualNStepQVals) -1)

            if(batchIndex[count][1] == True):
                actualSLMCQVals = np.full_like(actualSLMCQVals, expertActionQVal) + self.margin
                actualSLMCQVals[0, action] = expertActionQVal
            else:
                if(expertAction == Action.SPLIT.value and currentState[0][3] == False):
                    actualSLMCQVals[0, expertAction] = (np.amin(actualSLMCQVals) -1)
            
            self.altModel.fit([currentState, currentState, currentState], [actualDQVals, actualNStepQVals, actualSLMCQVals])
            count += 1

        self.updateTargetWeights(False)    
        self.explorationRate *= self.explorationDecay
        if(self.explorationRate < self.explorationMin):
            self.explorationRate = self.explorationMin"""