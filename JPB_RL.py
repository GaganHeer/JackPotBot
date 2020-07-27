# Actions are completely randomized

import time
import gym
import random as rand
import JPB_BJ as bj

env = bj.BlackjackEnv()
num_episodes = 10000
rewardList = []

for i in range(num_episodes):
    # Reset the blackjack env and draw a new hand
    state = env.reset()
    finished = False
    totalReward = 0
    handNum = 0
    numActions = 3
    if((len(state[3][handNum]) == 2) and state[3][handNum][0] == state[3][handNum][1]):
        numActions = 4
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
        # Pick a random action
        randomAction = rand.randrange(0, numActions)
        # Results after performing the action
        observation,reward,done,info = env.step(randomAction, handNum)
        
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
        
        # Check if any hand is not completed else keep looping
        keys = done.keys()
        for key in keys:
            if(done.get(key) == False):
                handNum = key
                finished = False
                break
            else:
                finished = True

        if((len(info[0][handNum]) == 2) and (info[0][handNum][0] == info[0][handNum][1])):
            numActions = 4
        else:
            numActions = 3

    for key in reward:
            totalReward += reward[key]
    print("=====================================")
    print('Episode', i,', Total reward:', totalReward)

env.close()
