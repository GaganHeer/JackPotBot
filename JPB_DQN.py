import random
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
from helpers.action import Action

class DQN():
    # Exploration Decay determines how quickly the agent stops exploring (closer to 0 means much quicker while closer to 1 means much slower)
    # Exploriation Min is the abolsute minimum exploriation rate that the agent can reach so that it doesn't completely stop exploration at any point
    # Exploration Rate determines how often the agent decides to explore and take alternate options
    # Discount Rate determines how valuable the agent thinks each reward is (closer to 0 means the current reward is more important while closer to 1 means that a future reward is more important)
    # Model is the structure of the neural network model that is being used to predict which actions will result in the greatest rewards
    def __init__(self, observationSpace, actionSpace, learnRate, discRate, explorationMin, explorationMax, explorationDecay):
        self.explorationDecay = explorationDecay
        self.explorationMin = explorationMin
        self.explorationRate = explorationMax
        self.discRate = discRate
        self.actionSpace = actionSpace
        self.batchSize = 20
        self.memory = deque(maxlen=10000)
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observationSpace,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.actionSpace, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learnRate))

    # Store information that can be sampled from at a later point
    def storeExperience(self, currentState, action, reward, nextObs, done):
        self.memory.append((currentState, action, reward, nextObs, done))

    # Decide an action to perform
    # Either random exploration or the best predicted result
    def decideAction(self, currentState):
        if np.random.rand() < self.explorationRate:
            if(currentState[0][3]):
                return random.randrange(self.actionSpace)
            else:
                return random.randrange(self.actionSpace - 1)
        QValues = self.model.predict(currentState)
        chosenAction = np.argmax(QValues[0])
        # If the split action is chosen, but the max number of hands are already in play then change the action to the next best option
        if(chosenAction == Action.SPLIT.value and currentState[0][3] == False):            
            options = set(QValues[0])
            options.remove(max(options))
            chosenAction = np.argmax(options)
        return chosenAction

    # Updates multiple experiences at once based on random sampling
    def experienceReplay(self):
        # Memory size is too small to pull from
        if len(self.memory) < self.batchSize:
            return
        # Pull random experiences of batchSize to sample from and update
        batch = random.sample(self.memory, self.batchSize)
        for currentState, action, reward, nextObs, done in batch:
            
            QUpdate = reward
            if not done:
                QUpdate = (reward + self.discRate * np.amax(self.model.predict(nextObs)[0]))
            QValues = self.model.predict(currentState)
            
            if(action == Action.SPLIT.value and currentState[0][3] == False):
                QValues[0][action] = None
            else:
                QValues[0][action] = QUpdate

            self.model.fit(currentState, QValues, verbose=0)
        self.explorationRate *= self.explorationDecay
        self.explorationRate = max(self.explorationMin, self.explorationRate)