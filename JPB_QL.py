import random

class QLearning():
    # Alpha factor determines the learning rate (closer to 0 means considering less info while closer to 1 means only considering more recent info)
    # Gamma factor determines the discount rate (closer to 0 means considering short term rewards while closer to 1 means considering long term rewards)
    # Epsilon determines the exploration rate
    # Initial Decay is the small decay factor used in the first 25% and last 25% of games
    # Middle Decay is the large decay factor used in the middle 50% of games
    def __init__(self, alphaFac, gammaFac, epsilon, numGames):
        self.alphaFac = alphaFac
        self.gammaFac = gammaFac        
        self.numGames = numGames
        self.numGamesLeft = numGames
        self.epsilon = epsilon
        self.initDecay = (0.25 * epsilon) / (0.25 * numGames)
        self.midDecay = (0.5 * epsilon) / (0.5 * numGames)
        self.QTable = dict()
        self.validActions = list(range(4))
        
    # Update alpha factor and epsilon value per action
    def updateParams(self):
        if self.numGamesLeft > 0.75 * self.numGames:
            self.epsilon -= self.initDecay
        elif self.numGamesLeft > 0.25 * self.numGames:
            self.epsilon -= self.midDecay
        elif self.numGamesLeft > 0:
            self.epsilon -= self.initDecay
        else:
            self.epsilon = 0.0
            self.alphaFac = 0.0
        self.numGamesLeft -= 1

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
        if random.random() > self.epsilon:
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
        self.updateParams()
        return action

    # Update QValue after performing an action
    def updateQValue(self, obs, action, reward, nextObs):
        # QValue = old QValue + learning rate * (reward + (discount rate * max QValue after action) - old QValue)
        self.QTable[obs][action] += self.alphaFac * (reward + (self.gammaFac * self.getMaxQValue(nextObs)) - self.QTable[obs][action])
    
    # Return entire QTable
    def getQTable(self):
        return self.QTable