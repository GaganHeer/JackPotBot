# Actions chosen by Deep Q Network (JPB_DQN)

import gym
import random as rand
import numpy as np
import pandas as pd
from prettytable import PrettyTable as pt
import JPB_BJ as bj
import JPB_DQN as dqn
from helpers import stats as st

env = bj.BlackjackEnv()
numGames = 1000
explorationMin = 0.01
explorationMax = 1.0
explorationDecay = 0.995
learnRate = 0.0001
discRate = 0.95
maxHands = 4
rewardList = []
observationSpace = len(env.observation_space)
actionSpace = env.action_space.n
agent = dqn.DQN(observationSpace, actionSpace, learnRate, discRate, explorationMin, explorationMax, explorationDecay)

for i in range(numGames):        
    # Reset the blackjack env and draw a new hand
    state = env.reset()
    finished = False
    totalReward = 0
    handNum = 0

    # Initial state of the game after resetting
    print("\n\n\n\n\n")
    print("---INITIAL HAND---")
    print("Player Cards:", state[3])
    print("Player Total:", state[2])
    print("Useable Aces:", state[1])
    print("Dealer Show Card:", state[0])
    print("\n")
    print("---GAME STARTING---")

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
        agent.storeExperience(currentState, action, reward[handNum], nextObs, done[handNum])
        state = observation

        # State of the game
        print("Player Cards:", info[0])
        print("Player Total:", observation[2])
        print("Dealer Cards:", info[1])
        print("Dealer Total:", info[2])
        print("Dealer Show Card:", observation[0])
        print("Useable Aces:", observation[1])
        print("Done: ", done)
        print("Reward: ", reward)
        
        keys = done.keys()
        for key in keys:
            if(done.get(key) == False):
                handNum = key
                finished = False
                break
            else:
                finished = True

        agent.experienceReplay()

    for key in reward:
        totalReward += reward[key]
    rewardList.append(totalReward)
    print("\n=====================================")
    print('Game', i,', Total reward:', totalReward)

# Display and export QTable
st.displayAndExportQTable(agent, observationSpace, actionSpace)
# Display average rewards
st.getAvgRewards(numGames, rewardList)
env.close()
