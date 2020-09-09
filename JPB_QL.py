import random

class QLearning():
    # Learning Rate determines how quickly the agent tries to learn (closer to 0 means considering less info while closer to 1 means only considering more recent info)
    # Discount Rate determines how valuable the agent thinks each reward is (closer to 0 means considering short term rewards while closer to 1 means considering long term rewards)
    # Exploration Rate determines how often the agent explores an alternate option
    def __init__(self, learnRate, discRate, explorationRate, numGames):
        self.learnRate = learnRate
        self.discRate = discRate        
        self.numGames = numGames
        self.explorationRate = explorationRate
        self.explorationDecay = 0.995
        self.explorationMin = 0.01
        self.QTable = dict()
        self.validActions = list(range(4))

    # Creates QValue if obs doesn't exist in QTable 
    # Set initial value to 0 for each possible action of the new QValue
    def createQValue(self, obs):
        if obs not in self.QTable:
            if(obs[3]):
                self.validActions = list(range(4))
            else:
                self.validActions = list(range(3))
            self.QTable[obs] = dict((action, 0.0) for action in self.validActions)

    # Returns the max Q value of the obs
    def getMaxQValue(self, obs):
        self.createQValue(obs)
        return max(self.QTable[obs].values())

    # Choose which action to perform
    # Either random exploration or highest QValue
    def decideAction(self, obs):
        self.createQValue(obs)

        # Choose action based on highest QValue
        if random.random() > self.explorationRate:
            maxQ = self.getMaxQValue(obs)
            maxQList = []
            for key in self.QTable[obs].keys():
                if self.QTable[obs][key] == maxQ:
                    maxQList.append(key)
            action = random.choice(maxQList)
        # Choose action based on random pick
        else:
            if(obs[3]):
                self.validActions = list(range(4))
            else:
                self.validActions = list(range(3))
            action = random.choice(self.validActions)
        self.explorationRate *= self.explorationDecay
        if(self.explorationRate < self.explorationMin):
            self.explorationRate = self.explorationMin
        return action

    # Update QValue after performing an action
    def updateQValue(self, obs, action, reward, nextObs):
        # QValue = old QValue + learning rate * (reward + (discount rate * max QValue after action) - old QValue)
        self.QTable[obs][action] += self.learnRate * (reward + (self.discRate * self.getMaxQValue(nextObs)) - self.QTable[obs][action])
    
    # Return entire QTable
    def getQTable(self):
        return self.QTable