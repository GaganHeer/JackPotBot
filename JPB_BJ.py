import gym
from gym import spaces
from gym.utils import seeding
import random as rand

# Four 10's because Jack, Queen, King all count as 10
cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

# Draw one card randomly from the deck
def drawCard():
    card = rand.choice(cards)
    return card

# Draw two cards randomly from the deck
def drawHand():
    hand = [[drawCard(), drawCard()]]
    #hand = {}
    #hand[0] = [drawCard(), drawCard()]
    return hand

# Determine if there is an ace in hand that can be used as an 11 without going over 21
def hasUseableAce(hand):
    if(1 in hand):
        if(sum(hand) + 10 <= 21):
            return True
    return False

# Determine the sum of hand
def handTotal(hand):
    if(hasUseableAce(hand)):    
        return sum(hand) + 10
    return sum(hand)

# Determine how strong the hand total is - over 21 is equivalent to 0, anything else is equivalent to itself
def handStrength(hand):
    if(handTotal(hand) > 21):
        return 0
    return handTotal(hand)

# Determine the reward that the player will recieve - if the player won then it will be a positive reward else a negative reward
def getReward(playerStrength, dealerStrength):
    reward = (playerStrength - dealerStrength)
    if(reward > 0):
        return 1
    elif (reward == 0):
        return 0
    else:
        return -1

class BlackjackEnv(gym.Env):

    def __init__(self):
        # 4 possible actions - stand (0), hit (1), double down (2), split (3)
        self.action_space = spaces.Discrete(4)
        # Player's hand total
        # Player has splittable hand
        # Dealer's upcard
        # Player has useable ace
        self.observation_space = spaces.Tuple((                       
            spaces.Discrete(20),
            spaces.Discrete(2),
            spaces.Discrete(10),
            spaces.Discrete(2)))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Stand action - end turn for that hand
    def standAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Stand")
        self.useableAces[handNum] = hasUseableAce(self.player[handNum])
        self.playerTotals[handNum] = handTotal(self.player[handNum])
        self.done[handNum] = True
        while handTotal(self.dealer[0]) < 17:
            self.dealer[0].append(drawCard())
        self.reward[handNum] = getReward(handStrength(self.player[handNum]), handStrength(self.dealer[0]))

    # Hit action - draw card for that hand
    def hitAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Hit")
        self.player[handNum].append(drawCard())
        self.useableAces[handNum] = hasUseableAce(self.player[handNum])
        self.playerTotals[handNum] = handTotal(self.player[handNum])
        if sum(self.player[handNum]) > 21:
            self.done[handNum] = True
            self.reward[handNum] = -1
        else:
            self.done[handNum] = False

    # Double down action - double your bet, draw one card and end your turn
    def doubleDownAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Double Down")
        self.done[handNum] = True
        self.player[handNum].append(drawCard())
        self.useableAces[handNum] = hasUseableAce(self.player[handNum])
        self.playerTotals[handNum] = handTotal(self.player[handNum])
        if sum(self.player[handNum]) > 21:
            self.reward[handNum] = -1
        else:
            while handTotal(self.dealer[0]) < 17:
                self.dealer[0].append(drawCard())
            self.reward[handNum] = (getReward(handStrength(self.player[handNum]), handStrength(self.dealer[0])) * 2)

    # Split action - if the first two cards in your hand are the same number 
    # then separate them into two new hands
    def splitAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Split")
        
        # Seperate cards into make two hands
        cardOne = self.player[handNum][0]
        cardTwo = self.player[handNum][1]
        self.player[handNum] = [cardOne, drawCard()]
        self.player.append([cardTwo, drawCard()])
        self.done[handNum] = False
        self.done[len(self.player) - 1] = False
        self.useableAces[handNum] = hasUseableAce(self.player[handNum])
        self.useableAces[len(self.player) - 1] = hasUseableAce(self.player[len(self.player) - 1])
        self.playerTotals[handNum] = handTotal(self.player[handNum])
        self.playerTotals[len(self.player) - 1] = handTotal(self.player[len(self.player) - 1])
        self.reward[handNum] = 0
        self.reward[len(self.player) - 1] = 0

    # Return information after taking an action
    def step(self, action, handNum):
        if(action == 0):
            self.standAction(handNum)
        elif(action == 1):
            self.hitAction(handNum)
        elif(action == 2):
            self.doubleDownAction(handNum)
        else:
            self.splitAction(handNum)

        return self.getObs(), self.reward, self.done, (self.player, self.dealer, handTotal(self.dealer[0]))

    # Return Dealer upcard, Player useable aces, Player total, Player cards
    def getObs(self):
        return (self.dealer[0][0], self.useableAces, self.playerTotals, self.player)

    # Reset environment
    def reset(self):
        self.dealer = drawHand()
        self.player = drawHand()
        self.reward = {}
        self.reward[0] = 0
        self.useableAces = {}
        self.useableAces[0] = hasUseableAce(self.player[0])
        self.playerTotals = {}
        self.playerTotals[0] = handTotal(self.player[0])
        self.done = {}
        return self.getObs()