import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random as rand

# Four 10's because Jack, Queen, King all count as 10
cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

# Draw one card randomly from the deck
def drawCard():
    """card = np.random.randint(1, len(cards)+1)
    if(card >= 10):
        card = 10"""
    card = rand.choice(cards)
    return card

# Draw two cards randomly from the deck
def drawHand():
    hand = [[drawCard(), drawCard()]]
    return hand

# Determine if there is an ace in hand that can be used as an 11 without going over 21
def hasUseableAce(hand):
    if(1 in hand):
        if(sum(hand) + 10 <= 21):
            return True
    return False

# Determine the sum of hand
def handTotal(hand):
    if hasUseableAce(hand):
        return sum(hand) + 10
    return sum(hand)

# Determine how strong the hand total is - over 21 is equivalent to 0, anything else is equivalent to itself
def handStrength(hand):
    if(handTotal(hand) > 21):
        return 0
    return handTotal(hand)

# Determine the reward that the player will recieve - if the player won then it will be a positive reward else a negative reward
def getReward(playerHand, dealerHand):
    reward = (playerHand - dealerHand)
    if(reward > 0):
        return 1
    else:
        return -1

class BlackjackEnv(gym.Env):

    def __init__(self):
        # 4 possible actions - stand (0), hit (1), double down (2), split (3)
        self.action_space = spaces.Discrete(4)

        # Player's maximum current sum
        # Dealer's upcard
        # Player has useable ace
        self.observation_space = spaces.Tuple((            
            spaces.Discrete(32),            
            spaces.Discrete(11),
            spaces.Discrete(2)))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def standAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Stand")
        self.done[handNum] = True
        while handTotal(self.dealer[0]) < 17:
            self.dealer[0].append(drawCard())
        self.reward += getReward(handStrength(self.player[handNum]), handStrength(self.dealer[0]))

    def hitAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Hit")
        self.player[handNum].append(drawCard())
        if sum(self.player[handNum]) > 21:
            self.done[handNum] = True
            self.reward += -1
        else:
            self.done[handNum] = False

    def doubleDownAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Double Down")
        self.done[handNum] = True
        self.player[handNum].append(drawCard())
        while handTotal(self.dealer[0]) < 17:
            self.dealer[0].append(drawCard())
        self.reward += (getReward(handStrength(self.player[handNum]), handStrength(self.dealer[0])) * 2)

    def splitAction(self, handNum):
        print("Hand Number: ", handNum)
        print("Action: Split")
        recursiveHandNum = 0
        cardOne = self.player[handNum][0]
        cardTwo = self.player[handNum][1]
        self.player[handNum] = [cardOne, drawCard()]
        self.player.append([cardTwo, drawCard()])
        self.done[handNum] = False
        self.done[len(self.player) - 1] = False

        print ("Player Cards:", self.player)
        print ("Dealer Cards:", self.dealer)
        print("Dealer Total:", handTotal(self.dealer[0]))
        print("Dealer Show Card:", self.dealer[0][0])
        print("Useable Aces:", hasUseableAce(self.player[0]))
        print("Reward: ", self.reward)
        print("Done: ", self.done)
        print("\n")

        #print("New Hands: ", self.player)
        #print("Dones: ", self.done)
        for hand in self.player:
            if(self.done[recursiveHandNum] == False):
                numActions = 3
                if(hand[0] == hand[1]):
                    self.canSplit = True
                    numActions += 1
                

                while not self.done[handNum]:
                    #print("Playing Hand: ", hand)
                    randomAction = rand.randrange(0, numActions)
                    if self.canSplit: numActions = 3    
                    self.canSplit =  False

                    if(randomAction == 0):
                        self.standAction(handNum)
                    elif(randomAction == 1):
                        self.hitAction(handNum)
                    elif(randomAction == 2):
                        self.doubleDownAction(handNum)
                    else:
                        self.splitAction(handNum)

                    print ("Player Cards:", self.player)
                    print ("Dealer Cards:", self.dealer)
                    print("Dealer Total:", handTotal(self.dealer[0]))
                    print("Dealer Show Card:", self.dealer[0][0])
                    print("Useable Aces:", hasUseableAce(self.player[0]))
                    print("Reward: ", self.reward)
                    print("Done: ", self.done)
                    print("")
                handNum += 1
            recursiveHandNum += 1

    def step(self, action):          
        return self.getObs(), self.reward, self.done, {}

    def getObs(self):
        # Dealer upcard, Player useable ace, Player hand, Dealer hand
        return (self.dealer[0][0], hasUseableAce(self.player[0]), self.player, self.dealer, handTotal(self.dealer[0]))

    def reset(self):
        self.dealer = drawHand()
        self.player = drawHand()
        self.canSplit = False
        self.reward = 0
        self.done = {}
        return self.getObs()