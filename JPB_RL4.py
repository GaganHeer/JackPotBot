# Actions chosen by Deep Q-Learning from Demonstrations (JPB_DQfD)

import gym
import random as rand
import numpy as np
import pandas as pd
import sys
from prettytable import PrettyTable as pt
import JPB_BJ as bj
import JPB_DQfD as dqfd
import JPB_Tests as tests
from helpers import stats as st
import datetime

def runTests(model, outputSize):
    testInput = (7, 0, 10, 0)
    testInput = np.reshape(testInput, [1, observationSpace])
    tests.runTestBed(model, testInput, actionSpace)

def initEnv():
    env = bj.BlackjackEnv()
    observationSpace = len(env.observation_space)
    actionSpace = env.action_space.n
    return env, observationSpace, actionSpace

def initAgent(observationSpace, actionSpace, learnRate, discRate, explorationMin, explorationMax, explorationDecay, nsteps, margin, batchSize, memSize, verbose):
    agent = dqfd.DQfD(observationSpace, actionSpace, learnRate, discRate, explorationMin, explorationMax, explorationDecay, nsteps, margin, batchSize, memSize, verbose)
    agent.createDQNModels()
    #runTests(agent.evalModel, actionSpace)
    agent.loadExpertData()
    agent.preTraining()
    #runTests(agent.evalModel, actionSpace)
    agent.loadTrainedModel()
    return agent

def displayInit(state):
    print("\n\n\n\n\n")
    print("---INITIAL HAND---")
    print("Player Cards:", state[3])
    print("Player Total:", state[2])
    print("Useable Aces:", state[1])
    print("Dealer Show Card:", state[0])
    print("\n")
    print("---GAME STARTING---")

def displayGame(info, observation, done, reward):
    print("Player Cards:", info[0])
    print("Player Total:", observation[2])
    print("Dealer Cards:", info[1])
    print("Dealer Total:", info[2])
    print("Dealer Show Card:", observation[0])
    print("Useable Aces:", observation[1])
    print("Done: ", done)
    print("Reward: ", reward, '\n')

def displayFinal(gameNum, totalReward):
    print("\n=====================================")
    print('Game', gameNum,', Total reward:', totalReward)

def checkHandStates(done):
    keys = done.keys()
    handNum = 0
    for key in keys:
        if(done.get(key) == False):
            handNum = key
            return False, handNum
        else:
            return True, handNum

def trackRewards(reward):
    totalReward = 0
    for key in reward:
        totalReward += reward[key]
    return totalReward

def saveFinalModel(numGames, modelName, agent):
    modelName = (str(numGames) + "_DQfD_Model.h5")
    agent.trainableModel.save_weights(modelName)
    modelNames.append(modelName)
    return modelNames

def createStats(agent, observationSpace, actionSpace, numGames, rewardList, startTime, endTime, modelNames):
    # Display and export QTable
    resultFile = st.displayAndExportQTable(agent, observationSpace, actionSpace, 'DQfD', numGames)
    # Calculates accuracy of model based on the action it takes in each possible state
    st.calcAccuracy(resultFile)
    # Save average rewards
    rewardFile = st.getAvgRewards(numGames, rewardList, startTime, endTime)
    # Plot average rewards per game
    st.plotRewards(rewardFile)
    # Plot accuracy over time
    st.plotAccuracy(agent, modelNames, numGames)

def runGames(numGames, env, agent, maxHands):
    rewardList = []
    modelNames = []

    for i in range(numGames):        
        # Reset the blackjack env and draw a new hand
        state = env.reset()
        finished = False
        totalReward = 0
        handNum = 0

        # Initial state of the game after resetting
        displayInit(state)

        if((i % int(numGames/10)) == 0):
            modelName = str(i) + "_DQfD_Model.h5"
            agent.trainableModel.save_weights(modelName)
            modelNames.append(modelName)

        while not finished:

            canSplit = False
            # Allow the option to split if the current hand is splittable and there are no more than 4 hands to play
            if((len(state[3][handNum]) == 2) and (state[3][handNum][0] == state[3][handNum][1]) and (len(state[3]) < maxHands)):
                canSplit = True

            # Seperate each hand that the player is playing into it's own state (Dealer show card, Current useable aces, Current hand total, Current ability to split)
            currentState = (state[0], state[1][handNum], state[2][handNum], canSplit)
            currentState = np.reshape(currentState, [1, observationSpace])

            action = agent.decideAction(currentState)
            observation,reward,done,info = env.step(action, handNum)

            canSplit = False
            if((len(observation[3][handNum]) == 2) and (observation[3][handNum][0] == observation[3][handNum][1]) and (len(observation[3]) < maxHands)):
                canSplit = True
            # Seperate each hand that the player is playing into it's own state
            nextObs = (observation[0], observation[1][handNum], observation[2][handNum], canSplit)
            nextObs = np.reshape(nextObs, [1, observationSpace])
            agent.storeExperience(agent.memoryBuffer, currentState, action, reward[handNum], nextObs, done[handNum], False)
            state = observation

            # State of the game
            displayGame(info, observation, done, reward)
            
            finished, handNum = checkHandStates(done)
            agent.experienceReplay()

        totalReward = trackRewards(reward)
        rewardList.append(totalReward)
        displayFinal(i, totalReward)

    return rewardList, modelNames, agent


if __name__ == "__main__":
    startTime = str(datetime.datetime.now())
    numGames = 10000
    explorationMin = 0.01
    explorationMax = 1.0
    explorationDecay = 0.995
    learnRate = 0.0001
    discRate = 0.95
    nsteps = 5
    batchSize = 20
    memSize = 20000
    margin = -0.7
    maxHands = 4
    verbose = 0
    env, observationSpace, actionSpace = initEnv()

    #agent = initAgent(observationSpace, actionSpace, learnRate, discRate, explorationMin, explorationMax, explorationDecay, nsteps, margin, batchSize, memSize, verbose)
    #rewardList, modelNames, agent = runGames(numGames, env, agent, maxHands)
    #modelNames = saveFinalModel(numGames, modelNames, agent)
    
    endTime = str(datetime.datetime.now())
    
    agent = dqfd.DQfD(observationSpace, actionSpace, learnRate, discRate, explorationMin, explorationMax, explorationDecay, nsteps, margin, batchSize, memSize, verbose)
    agent.loadTrainedModel(True, "10000_DQfD_Model.h5")
    modelNames = ["0_DQfD_Model.h5", "1000_DQfD_Model.h5", "2000_DQfD_Model.h5", "3000_DQfD_Model.h5", "4000_DQfD_Model.h5", "5000_DQfD_Model.h5", "6000_DQfD_Model.h5", "7000_DQfD_Model.h5", "8000_DQfD_Model.h5", "9000_DQfD_Model.h5", "10000_DQfD_Model.h5"]
    rewardList = []

    #runTests(agent.trainableModel, actionSpace)
    createStats(agent, observationSpace, actionSpace, numGames, rewardList, startTime, endTime, modelNames)
    env.close()