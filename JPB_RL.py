import time
import gym
import random as rand
import JPB_BJ as bj

env = bj.BlackjackEnv()
num_episodes = 20
done = False

for i in range(num_episodes):
    # Reset the blackjack env and draw a new hand
    state = env.reset()
    finished = False
    totalReward = 0
    handNum = 0
    canSplit = False
    hasSplit = False
    if(state[2][handNum][0] == state[2][handNum][1]): canSplit = True
    
    # Initial state of the game after resetting
    print("\n\n\n\n\n")
    print("---INITIAL HAND---")
    print ("Player Cards:", state[2])
    print ("Dealer Cards:", state[3])
    print("Dealer Total:", state[4])
    print("Dealer Show Card:", state[0])
    print("Useable Aces:", state[1])
    print("\n")
    print("---GAME STARTING---")

    while not finished:
        # Pick a random action
        # If splitting is available then allow that option
        numActions = 3
        if canSplit: numActions = 4
        randomAction = rand.randrange(0, numActions)
        # Splitting is only available on first action of hand
        canSplit = False
        
        # Perform the chosen action
        if(randomAction == 0):
            env.standAction(handNum)
        elif(randomAction == 1):
            env.hitAction(handNum)
        elif(randomAction == 2):
            env.doubleDownAction(handNum)
        else:
            hasSplit = True
            env.splitAction(handNum)

        # Results after performing the action
        observation,reward,done,info = env.step(randomAction)
        if not hasSplit:
            print ("Player Cards:", observation[2])
            print ("Dealer Cards:", observation[3])
            print("Dealer Total:", observation[4])
            print("Dealer Show Card:", observation[0])
            print("Useable Aces:", observation[1])
            print("Done: ", done)
            print("Reward: ", reward)
            print("")
        totalReward += reward
        
        # Check if any hand is not completed else keep looping
        keys = done.keys()
        for key in keys:
            if(done.get(key) == False):
                handNum = key
                finished = False
                break
            else:
                finished = True

    print("=====================================")
    print('Episode', i,', Total reward:', totalReward)

env.close()
