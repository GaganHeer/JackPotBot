# Stats class for sorting data points and creating tables, graphs etc...

import numpy as np
import pandas as pd
from prettytable import PrettyTable as pt
from helpers.action import Action

# All possible states
ace2 = [None] * 10
ace3 = [None] * 10
ace4 = [None] * 10
ace5 = [None] * 10
ace6 = [None] * 10
ace7 = [None] * 10
ace8 = [None] * 10
ace9 = [None] * 10
twinAce = [None] * 10
twin2 = [None] * 10
twin3 = [None] * 10
twin4 = [None] * 10
twin5 = [None] * 10
twin6 = [None] * 10
twin7 = [None] * 10
twin8 = [None] * 10
twin9 = [None] * 10
twin10 = [None] * 10
total5 = [None] * 10
total6 = [None] * 10
total7 = [None] * 10
total8 = [None] * 10
total8 = [None] * 10
total9 = [None] * 10
total10 = [None] * 10
total11 = [None] * 10
total12 = [None] * 10
total13 = [None] * 10
total14 = [None] * 10
total15 = [None] * 10
total16 = [None] * 10
total17 = [None] * 10
total18 = [None] * 10
total19 = [None] * 10
total20 = [None] * 10
total21 = [None] * 10

def getAvgRewards(numGames, rewardList):
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

def getBestAction(QValues, splittable):
    indexOfBestAction = np.argmax(QValues[0])
    if(not splittable and indexOfBestAction == Action.SPLIT.value):
        options = set(QValues[0])
        options.remove(max(options))
        indexOfBestAction = np.argmax(options)

    if(indexOfBestAction == Action.STAND.value):
        bestAction = "S"
    elif(indexOfBestAction == Action.HIT.value):
        bestAction = "H"
    elif(indexOfBestAction == Action.DOUBLE_DOWN.value):
        bestAction = "D"
    elif(indexOfBestAction == Action.SPLIT.value):
        bestAction = "P"
    else:
        raise Exception("Index of best action is out of range")
    return bestAction

def createDF(fileName):
    # Full list of possible states
    tempList = [total5, total6, total7, total8, total9, total10, total11, total12, total13, total14, total15, total16, total17, total18, total19, total20, total21,
                ace2, ace3, ace4, ace5, ace6, ace7, ace8, ace9,
                twin2, twin3, twin4, twin5, twin6, twin7, twin8, twin9, twin10, twinAce]
    # Dataframe creation of QValues with column names set to dealer card states
    df = pd.DataFrame(tempList, columns = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # Changing row names to player card states
    df.rename(index={0:'5', 1: '6', 2: '7', 3: '8', 4: '9', 5: '10', 6: '11', 7: '12', 8: '13', 9: '14', 10: '15', 11: '16', 12: '17', 13: '18', 14: '19', 15: '20', 16: '21',
                        17:'A,2', 18: 'A,3', 19: 'A,4', 20: 'A,5', 21: 'A,6', 22: 'A,7', 23: 'A,8', 24: 'A,9',
                        25:'2,2', 26: '3,3', 27: '4,4', 28: '5,5', 29: '6,6', 30: '7,7', 31: '8,8', 32: '9,9', 33: '10,10', 34: 'A,A'}, inplace=True)
    df.to_csv(fileName + '_Results.csv')

def displayQTable(agent):
    print("\nQ TABLE")
    t = pt(['Player Total', 'Can Split', 'Useable Ace', 'Dealer Upcard', 'Stand Reward', 'Hit Reward', 'Double Down Reward', 'Split Reward'])
    qtable = agent.getQTable()
    for key in qtable:
        if(len(qtable[key]) == 4):
            t.add_row([key[2], key[3], key[1], key[0], qtable[key][0], qtable[key][1], qtable[key][2], qtable[key][3]])
        else:
            t.add_row([key[2], key[3], key[1], key[0], qtable[key][0], qtable[key][1], qtable[key][2], 'N/A'])
    print(t)
    print("")


def exportQTable(agent):
    qtable = agent.getQTable()
    for key in qtable:
        playerTotal = key[2]
        dealerCard = key[0]

        # Prediction for hands that are splittable
        if(playerTotal >= 4 and playerTotal <= 20 and len(qtable[key]) == 4):
            QValues = [[qtable[key][0], qtable[key][1], qtable[key][2], qtable[key][3]]]
            bestAction = getBestAction(QValues, True)

            if(playerTotal == 12 and key[1]):
                twinAce[dealerCard - 1] = bestAction
            elif(playerTotal == 4):
                twin2[dealerCard - 1] = bestAction
            elif(playerTotal == 6):
                twin3[dealerCard - 1] = bestAction
            elif(playerTotal == 8):
                twin4[dealerCard - 1] = bestAction
            elif(playerTotal == 10):
                twin5[dealerCard - 1] = bestAction
            elif(playerTotal == 12):
                twin6[dealerCard - 1] = bestAction
            elif(playerTotal == 14):
                twin7[dealerCard - 1] = bestAction
            elif(playerTotal == 16):
                twin8[dealerCard - 1] = bestAction
            elif(playerTotal == 18):
                twin9[dealerCard - 1] = bestAction
            elif(playerTotal == 20):
                twin10[dealerCard - 1] = bestAction

        # Prediction for hands that have useable aces
        elif(playerTotal >= 13 and playerTotal <= 20 and key[1]):
            QValues = [[qtable[key][0], qtable[key][1], qtable[key][2], -99]]
            bestAction = getBestAction(QValues, False)

            if(playerTotal == 13):
                ace2[dealerCard - 1] = bestAction
            elif(playerTotal == 14):
                ace3[dealerCard - 1] = bestAction
            elif(playerTotal == 15):
                ace4[dealerCard - 1] = bestAction
            elif(playerTotal == 16):
                ace5[dealerCard - 1] = bestAction
            elif(playerTotal == 17):
                ace6[dealerCard - 1] = bestAction
            elif(playerTotal == 18):
                ace7[dealerCard - 1] = bestAction
            elif(playerTotal == 19):
                ace8[dealerCard - 1] = bestAction
            elif(playerTotal == 20):
                ace9[dealerCard - 1] = bestAction
        
        elif(playerTotal >= 5 and playerTotal <= 21):
            QValues = [[qtable[key][0], qtable[key][1], qtable[key][2], -99]]
            bestAction = getBestAction(QValues, False)

            if(playerTotal == 5):
                total5[dealerCard - 1] = bestAction
            elif(playerTotal == 6):
                total6[dealerCard - 1] = bestAction
            elif(playerTotal == 7):
                total7[dealerCard - 1] = bestAction
            elif(playerTotal == 8):
                total8[dealerCard - 1] = bestAction
            elif(playerTotal == 9):
                total9[dealerCard - 1] = bestAction
            elif(playerTotal == 10):
                total10[dealerCard - 1] = bestAction
            elif(playerTotal == 11):
                total11[dealerCard - 1] = bestAction
            elif(playerTotal == 12):
                total12[dealerCard - 1] = bestAction
            elif(playerTotal == 13):
                total13[dealerCard - 1] = bestAction
            elif(playerTotal == 14):
                total14[dealerCard - 1] = bestAction
            elif(playerTotal == 15):
                total15[dealerCard - 1] = bestAction
            elif(playerTotal == 16):
                total16[dealerCard - 1] = bestAction
            elif(playerTotal == 17):
                total17[dealerCard - 1] = bestAction
            elif(playerTotal == 18):
                total18[dealerCard - 1] = bestAction
            elif(playerTotal == 19):
                total19[dealerCard - 1] = bestAction
            elif(playerTotal == 20):
                total20[dealerCard - 1] = bestAction
            elif(playerTotal == 21):
                total21[dealerCard - 1] = bestAction

    createDF('QL')

def displayAndExportQTable(agent, observationSpace, actionSpace):

    print("\nQ TABLE")
    t = pt(['Player Total', 'Can Split', 'Useable Ace', 'Dealer Upcard', 'Stand Reward', 'Hit Reward', 'Double Down Reward', 'Split Reward'])
    # Dealer Card (1 to 10)
    for dealerCard in range(1, 11):
        # Player Total (2 to 21)
        for playerTotal in range(2, 22):
            
            # Prediction for hands that are splittable
            if(playerTotal % 2 == 0 and playerTotal >= 4 and playerTotal <= 20):
                splitState = (dealerCard, False, playerTotal, True)
                splitState = np.reshape(splitState, [1, observationSpace])
                QValues = agent.model.predict(splitState)
                t.add_row([playerTotal, 'True', 'False', dealerCard, QValues[0][0], QValues[0][1], QValues[0][2], QValues[0][3]])
                bestAction = getBestAction(QValues, True)

                if(playerTotal == 4):
                    twin2[dealerCard - 1] = bestAction
                elif(playerTotal == 6):
                    twin3[dealerCard - 1] = bestAction
                elif(playerTotal == 8):
                    twin4[dealerCard - 1] = bestAction
                elif(playerTotal == 10):
                    twin5[dealerCard - 1] = bestAction
                elif(playerTotal == 12):
                    twin6[dealerCard - 1] = bestAction
                elif(playerTotal == 14):
                    twin7[dealerCard - 1] = bestAction
                elif(playerTotal == 16):
                    twin8[dealerCard - 1] = bestAction
                elif(playerTotal == 18):
                    twin9[dealerCard - 1] = bestAction
                elif(playerTotal == 20):
                    twin10[dealerCard - 1] = bestAction

            # Prediction for hands that have useable aces
            if(playerTotal >= 13 and playerTotal <= 20):
                aceState = (dealerCard, True, playerTotal, False)
                aceState = np.reshape(aceState, [1, observationSpace])
                QValues = agent.model.predict(aceState)
                t.add_row([playerTotal, 'False', 'True', dealerCard, QValues[0][0], QValues[0][1], QValues[0][2], 'N/A'])
                bestAction = getBestAction(QValues, False)

                if(playerTotal == 13):
                    ace2[dealerCard - 1] = bestAction
                elif(playerTotal == 14):
                    ace3[dealerCard - 1] = bestAction
                elif(playerTotal == 15):
                    ace4[dealerCard - 1] = bestAction
                elif(playerTotal == 16):
                    ace5[dealerCard - 1] = bestAction
                elif(playerTotal == 17):
                    ace6[dealerCard - 1] = bestAction
                elif(playerTotal == 18):
                    ace7[dealerCard - 1] = bestAction
                elif(playerTotal == 19):
                    ace8[dealerCard - 1] = bestAction
                elif(playerTotal == 20):
                    ace9[dealerCard - 1] = bestAction

            normalState = (dealerCard, False, playerTotal, False)
            normalState = np.reshape(normalState, [1, observationSpace])
            QValues = agent.model.predict(normalState)
            t.add_row([playerTotal, 'False', 'False', dealerCard, QValues[0][0], QValues[0][1], QValues[0][2], 'N/A'])
            bestAction = getBestAction(QValues, False)

            if(playerTotal == 5):
                total5[dealerCard - 1] = bestAction
            elif(playerTotal == 6):
                total6[dealerCard - 1] = bestAction
            elif(playerTotal == 7):
                total7[dealerCard - 1] = bestAction
            elif(playerTotal == 8):
                total8[dealerCard - 1] = bestAction
            elif(playerTotal == 9):
                total9[dealerCard - 1] = bestAction
            elif(playerTotal == 10):
                total10[dealerCard - 1] = bestAction
            elif(playerTotal == 11):
                total11[dealerCard - 1] = bestAction
            elif(playerTotal == 12):
                total12[dealerCard - 1] = bestAction
            elif(playerTotal == 13):
                total13[dealerCard - 1] = bestAction
            elif(playerTotal == 14):
                total14[dealerCard - 1] = bestAction
            elif(playerTotal == 15):
                total15[dealerCard - 1] = bestAction
            elif(playerTotal == 16):
                total16[dealerCard - 1] = bestAction
            elif(playerTotal == 17):
                total17[dealerCard - 1] = bestAction
            elif(playerTotal == 18):
                total18[dealerCard - 1] = bestAction
            elif(playerTotal == 19):
                total19[dealerCard - 1] = bestAction
            elif(playerTotal == 20):
                total20[dealerCard - 1] = bestAction
            elif(playerTotal == 21):
                total21[dealerCard - 1] = bestAction
            
        # Prediction for a hand that has double aces
        doubleAceState = (dealerCard, True, 12, True)
        doubleAceState = np.reshape(doubleAceState, [1, observationSpace])
        QValues = agent.model.predict(doubleAceState)
        t.add_row([12, 'True', 'True', dealerCard, QValues[0][0], QValues[0][1], QValues[0][2], QValues[0][3]])
        bestAction = getBestAction(QValues, True)
        twinAce[dealerCard - 1] = bestAction

    print(t)
    print("")
    createDF('DQN')