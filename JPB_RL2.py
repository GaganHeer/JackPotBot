# Actions are based on QTable (JPB_QL)

import gym
import random as rand
import datetime
from prettytable import PrettyTable as pt
import JPB_BJ as bj
import JPB_QL as ql
from helpers import stats as st

startTime = str(datetime.datetime.now())
env = bj.BlackjackEnv()
numGames = 10000
explorationRate = 1.0
learnRate = 0.1
discRate = 0.995
rewardList = []
agent = ql.QLearning(learnRate, discRate, explorationRate, numGames)

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
        # Check if hand is splittable
        canSplit = False
        if((len(state[3][handNum]) == 2) and state[3][handNum][0] == state[3][handNum][1]):
            canSplit = True
        # Seperate each hand that the player is playing into it's own state
        currentState = (state[0], state[1][handNum], state[2][handNum], canSplit)
        # Pick a random action
        randomAction = agent.decideAction(currentState)
        # Results after performing the action
        observation,reward,done,info = env.step(randomAction, handNum)
        # Check if next hand is splittable
        canSplit = False
        if((len(observation[3][handNum]) == 2) and observation[3][handNum][0] == observation[3][handNum][1]):
            canSplit = True
        # Seperate each hand that the player is playing into it's own state
        nextObs = (observation[0], observation[1][handNum], observation[2][handNum], canSplit)
        # Pass player's state, action, reward and next state to adjust q-table
        agent.updateQValue(currentState, randomAction, reward[handNum], nextObs)
        # Assign the observation after performing an action as the new state
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
        print("")
        
        # Check if every hand has completed playing else keep looping
        keys = done.keys()
        for key in keys:
            if(done.get(key) == False):
                handNum = key
                finished = False
                break
            else:
                finished = True

    # Add up and display total reward for the game
    for key in reward:
        totalReward += reward[key]
    rewardList.append(totalReward)
    print("=====================================")
    print('Game', i,', Total reward:', totalReward)

endTime = str(datetime.datetime.now())
# Display QTable
st.displayQTable(agent)
# Export QTable
resultFile = st.exportQTable(agent, numGames)
# Display average rewards
rewardFile = st.getAvgRewards(numGames, rewardList, startTime, endTime)
# Calculates accuracy of model based on the action it takes in each possible state
st.calcAccuracy(resultFile)
env.close()
