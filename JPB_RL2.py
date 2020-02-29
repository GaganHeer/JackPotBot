import time
import gym
import random as rand
import JPB_BJ as bj
import JPB_QL as ql
from prettytable import PrettyTable as pt

env = bj.BlackjackEnv()
numGames = 10000
epsilon = 1.0
alphaFac = 0.8
gammaFac = 0.9
rewardList = []


agent = ql.QLearning(alphaFac, gammaFac, epsilon, numGames)

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
        if((len(state[3][handNum]) == 2) and state[3][handNum][0] == state[3][handNum][1]):
            canSplit = True
        # Seperate each hand that the player is playing into it's own state
        currentState = (state[0], state[1][handNum], state[2][handNum], canSplit)
        
        # Pick a random action
        randomAction = agent.decideAction(currentState)
        # Results after performing the action
        observation,reward,done,info = env.step(randomAction, handNum)

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

    for key in reward:
        totalReward += reward[key]
    rewardList.append(totalReward)
    print("=====================================")
    print('Game', i,', Total reward:', totalReward)

print("")
print("Q TABLE")
t = pt(['Player Total', 'Can Split', 'Useable Ace', 'Dealer Upcard', 'Stand Reward', 'Hit Reward', 'Double Down Reward', 'Split Reward'])
qtable = agent.getQTable()
for key in qtable:
    if(len(qtable[key]) == 4):
        t.add_row([key[2], key[3], key[1], key[0], qtable[key][0], qtable[key][1], qtable[key][2], qtable[key][3]])
    else:
        t.add_row([key[2], key[3], key[1], key[0], qtable[key][0], qtable[key][1], qtable[key][2], 'N/A'])
print(t)
print("")

splitAmt = 10
episodeSplit = (numGames/splitAmt)
i = 0
while(i < splitAmt):
    i += 1
    summingRange = int(i * episodeSplit)
    avg = (float(sum(rewardList[0:summingRange]))/summingRange)
    print("TOTAL REWARDS AFTER", summingRange, "GAMES:", sum(rewardList[0:summingRange]))
    print("AVG REWARDS AFTER", summingRange, "GAMES:", avg)
    print("")

env.close()
