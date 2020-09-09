# Stats class for sorting data points and creating tables, graphs etc...

import os
import csv
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt
from helpers.action import Action

# All possible states to be filled with predicted actions
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

# All possible states with expert actions
expAce2 = ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
expAce3 = ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
expAce4 = ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
expAce5 = ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
expAce6 = ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
expAce7 = ['S', 'D', 'D', 'D', 'D', 'S', 'S', 'H', 'H', 'H']
expAce8 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
expAce9 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
expTwin2 = ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H']
expTwin3 = ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H']
expTwin4 = ['H', 'H', 'H', 'P', 'P', 'H', 'H', 'H', 'H', 'H']
expTwin5 = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H']
expTwin6 = ['P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H', 'H']
expTwin7 = ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H']
expTwin8 = ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']
expTwin9 = ['P', 'P', 'P', 'P', 'P', 'S', 'P', 'P', 'S', 'S']
expTwin10 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
expTwinAce = ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']
expTotal5 = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
expTotal6 = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
expTotal7 = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
expTotal8 = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
expTotal9 = ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
expTotal10 = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H']
expTotal11 = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H']
expTotal12 = ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
expTotal13 = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
expTotal14 = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
expTotal15 = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
expTotal16 = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
expTotal17 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
expTotal18 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
expTotal19 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
expTotal20 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
expTotal21 = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']

def getAvgRewards(numGames, rewardList, startTime=0, endTime=0):
    splitAmt = 10
    episodeSplit = (numGames/splitAmt)
    i = 0
    avg = 0
    fileName = "Reward Totals.txt"
    if os.path.exists(fileName):
        os.remove(fileName)
    text_file = open(fileName, "a")
    output = []
    output.append("START TIME: ")
    output.append(startTime)
    output.append("\n")
    output.append("END TIME: ")
    output.append(endTime)
    output.append("\n\n\n")
    s = ''.join(output)
    text_file.write(s)
    while(i < splitAmt):
        i += 1
        summingRange = int(i * episodeSplit)
        avg = (float(sum(rewardList[0:summingRange]))/summingRange)
        print("TOTAL REWARDS AFTER", summingRange, "GAMES:", sum(rewardList[0:summingRange]))
        print("AVG REWARDS AFTER", summingRange, "GAMES:", avg)
        print("")
        output = []
        output.append("GAMES:")
        output.append(str(summingRange))
        output.append("\nTOTAL REWARDS:")
        output.append(str(sum(rewardList[0:summingRange])))
        output.append("\nAVG REWARDS:")
        output.append(str(avg))
        output.append("\n\n\n")
        s = ''.join(output)
        text_file.write(s)
    text_file.close()
    return fileName
        
def plotRewards(fileName):
    games = []
    avgRewards = []
    rewardFile = open(fileName, "r")
    for line in rewardFile:
        if("GAMES" in line):
            templine = float((line.partition(":")[2])[:-1])
            games.append(templine)
        if("AVG" in line):
            templine = float((line.partition(":")[2])[:-1])
            avgRewards.append(templine)
    
    DQNrewards = [-0.321, -0.3175, -0.296, -0.29, -0.27, -0.27889, -0.2716, -0.263625, -0.24278, -0.2436]
    QLrewards = [-0.238, -0.217, -0.235, -0.23625, -0.2342, -0.24016, -0.25, -0.243125, -0.26788, -0.2597]

    plt.plot(games, avgRewards, label="DQfD")
    plt.plot(games, DQNrewards, label="DQN")
    plt.plot(games, QLrewards, label="QLearning")

    totalMin = min(avgRewards)
    if(min(DQNrewards) < totalMin):
        totalMin = min(DQNrewards)
    if(min(QLrewards) < totalMin):
        totalMin = min(QLrewards)

    totalMax = max(avgRewards)
    if(max(DQNrewards) > totalMax):
        totalMax = max(DQNrewards)
    if(max(QLrewards) > totalMax):
        totalMax = max(QLrewards)

    plt.xlim(games[0], games[-1])
    plt.ylim(totalMin, totalMax)
    plt.xlabel("Number of Games")
    plt.ylabel("Average Rewards")
    plt.legend()
    plt.savefig("Reward Graph", dpi=300)
    #plt.show()

def plotAccuracy(agent, modelNames, numGames=10000, splitAmt=10):
    games = []
    acc = []
    correctActions = []
    splitAfter = numGames/splitAmt
    for x in range(splitAmt+1):
        games.append(numGames * (x/10))
    
    for name in modelNames:
        agent.loadTrainedModel(True, name)
        resultFile = displayAndExportQTable(agent, 4, 4, 'DQfD', numGames)
        tempAcc, tempCorrects = calcAccuracy(resultFile)
        tempAcc = tempAcc*100
        acc.append(tempAcc)
        correctActions.append(tempCorrects)

    plt.clf()
    plt.plot(games, acc, label="DQfD")
    plt.xlim(games[0], games[-1])
    plt.ylim(0, 100)
    plt.xlabel("Number of Games")
    plt.ylabel("Accuracy")
    plt.savefig("Accuracy Graph", dpi=300)
    #plt.show()

    plt.clf()
    lenGames = np.arange(len(games))
    plt.bar(lenGames, correctActions, align='center')
    plt.xticks(lenGames, games)
    plt.xlabel("Number of Games")
    plt.ylabel("Number of Correct Predictions")
    plt.savefig("Correct Predictions Graph", dpi=300)
    #plt.show()

    """plt.plot(games, correctActions, label="DQfD")
    plt.xlim(games[0], games[-1])
    plt.ylim(0, 350)
    plt.xlabel("Number of Games")
    plt.ylabel("Number of Correct Predictions")
    plt.legend()
    plt.savefig("Correct Predictions Graph", dpi=300)
    #plt.show()"""

def getBestAction(QValues, splittable):
    indexOfBestAction = np.argmax(QValues[0])
    if(not splittable and indexOfBestAction == Action.SPLIT.value):
        newQValues = [QValues[0][0], QValues[0][1], QValues[0][2]]
        indexOfBestAction = np.argmax(newQValues)

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
    #tempList = [total5, total6, total7, total8, total9, total10, total11, total12, total13, total14, total15, total16, total17, total18, total19, total20, total21,
    #            ace2, ace3, ace4, ace5, ace6, ace7, ace8, ace9,
    #            twin2, twin3, twin4, twin5, twin6, twin7, twin8, twin9, twin10, twinAce]

    tempList = [twinAce, twin10, twin9, twin8, twin7, twin6, twin5, twin4, twin3, twin2,
                ace9, ace8, ace7, ace6, ace5, ace4, ace3, ace2,
                total21, total20, total19, total18, total17, total16, total15, total14, total13, total12, total11, total10, total9, total8, total7, total6, total5]
    # Dataframe creation of QValues with column names set to dealer card states
    df = pd.DataFrame(tempList, columns = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # Changing row names to player card states
    #df.rename(index={0:'5', 1: '6', 2: '7', 3: '8', 4: '9', 5: '10', 6: '11', 7: '12', 8: '13', 9: '14', 10: '15', 11: '16', 12: '17', 13: '18', 
    #                    14: '19', 15: '20', 16: '21', 17:'A,2', 18: 'A,3', 19: 'A,4', 20: 'A,5', 21: 'A,6', 22: 'A,7', 23: 'A,8', 24: 'A,9',
    #                    25:'2,2', 26: '3,3', 27: '4,4', 28: '5,5', 29: '6,6', 30: '7,7', 31: '8,8', 32: '9,9', 33: '10,10', 34: 'A,A'}, inplace=True)

    df.rename(index={0:'A,A', 1: '10,10', 2: '9,9', 3: '8,8', 4: '7,7', 5: '6,6', 6: '5,5', 7: '4,4', 8: '3,3', 9: '2,2', 10: 'A,9', 11: 'A,8', 
                        12: 'A,7', 13: 'A,6', 14: 'A,5', 15: 'A,4', 16: 'A,3', 17:'A,2', 18: '21', 19: '20', 20: '19', 21: '18', 22: '17', 
                        23: '16', 24: '15', 25:'14', 26: '13', 27: '12', 28: '11', 29: '10', 30: '9', 31: '8', 32: '7', 33: '6', 34:'5'}, inplace=True)
    df.to_csv(fileName)

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

def calcAccuracy(fileName):
    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        resultsList = list(reader)
    resultsList.pop(0)
    
    for i in range(len(resultsList)):
        resultsList[i].pop(0) 

    total5 = resultsList[34]
    total6 = resultsList[33]
    total7 = resultsList[32]
    total8 = resultsList[31]
    total9 = resultsList[30]
    total10 = resultsList[29]
    total11 = resultsList[28]
    total12 = resultsList[27]
    total13 = resultsList[26]
    total14 = resultsList[25]
    total15 = resultsList[24]
    total16 = resultsList[23]
    total17 = resultsList[22]
    total18 = resultsList[21]
    total19 = resultsList[20]
    total20 = resultsList[19]
    total21 = resultsList[18]
    ace2 = resultsList[17]
    ace3 = resultsList[16]
    ace4 = resultsList[15]
    ace5 = resultsList[14]
    ace6 = resultsList[13]
    ace7 = resultsList[12]
    ace8 = resultsList[11]
    ace9 = resultsList[10]
    twin2 = resultsList[9]
    twin3 = resultsList[8]
    twin4 = resultsList[7]
    twin5 = resultsList[6]
    twin6 = resultsList[5]
    twin7 = resultsList[4]
    twin8 = resultsList[3]
    twin9 = resultsList[2]
    twin10 = resultsList[1]
    twinAce = resultsList[0]

    correctActions = 0
    totalActions = 350

    for i in range(len(total5)):
        if(expTotal5[i] == total5[i]):
            correctActions += 1

        if(expTotal6[i] == total6[i]):
            correctActions += 1

        if(expTotal7[i] == total7[i]):
            correctActions += 1

        if(expTotal8[i] == total8[i]):
            correctActions += 1

        if(expTotal9[i] == total9[i]):
            correctActions += 1

        if(expTotal10[i] == total10[i]):
            correctActions += 1

        if(expTotal11[i] == total11[i]):
            correctActions += 1

        if(expTotal12[i] == total12[i]):
            correctActions += 1

        if(expTotal13[i] == total13[i]):
            correctActions += 1

        if(expTotal14[i] == total14[i]):
            correctActions += 1

        if(expTotal15[i] == total15[i]):
            correctActions += 1

        if(expTotal16[i] == total16[i]):
            correctActions += 1

        if(expTotal17[i] == total17[i]):
            correctActions += 1

        if(expTotal18[i] == total18[i]):
            correctActions += 1

        if(expTotal19[i] == total19[i]):
            correctActions += 1

        if(expTotal20[i] == total20[i]):
            correctActions += 1

        if(expTotal21[i] == total21[i]):
            correctActions += 1
            
        if(expAce2[i] == ace2[i]):
            correctActions += 1

        if(expAce3[i] == ace3[i]):
            correctActions += 1

        if(expAce4[i] == ace4[i]):
            correctActions += 1

        if(expAce5[i] == ace5[i]):
            correctActions += 1

        if(expAce6[i] == ace6[i]):
            correctActions += 1

        if(expAce7[i] == ace7[i]):
            correctActions += 1

        if(expAce8[i] == ace8[i]):
            correctActions += 1

        if(expAce9[i] == ace9[i]):
            correctActions += 1

        if(expTwin2[i] == twin2[i]):
            correctActions += 1

        if(expTwin3[i] == twin3[i]):
            correctActions += 1

        if(expTwin4[i] == twin4[i]):
            correctActions += 1

        if(expTwin5[i] == twin5[i]):
            correctActions += 1

        if(expTwin6[i] == twin6[i]):
            correctActions += 1
        
        if(expTwin7[i] == twin7[i]):
            correctActions += 1

        if(expTwin8[i] == twin8[i]):
            correctActions += 1

        if(expTwin9[i] == twin9[i]):
            correctActions += 1

        if(expTwin10[i] == twin10[i]):
            correctActions += 1

        if(expTwinAce[i] == twinAce[i]):
            correctActions += 1
    
    acc = correctActions/totalActions
    #print(correctActions)
    #print(acc)
    return acc, correctActions

def exportQTable(agent, numGames):
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

    fileName = 'Results_' + str(int(numGames/1000)) + 'K_Games.csv'
    createDF(fileName)
    return fileName

def displayAndExportQTable(agent, observationSpace, actionSpace, name, numGames):

    print("\nQ TABLE")
    t = pt(['Player Total', 'Can Split', 'Useable Ace', 'Dealer Upcard', 'Stand Reward', 'Hit Reward', 'Double Down Reward', 'Split Reward'])
    QValues = None
    # Dealer Card (1 to 10)
    for dealerCard in range(1, 11):
        # Player Total (2 to 21)
        for playerTotal in range(2, 22):

            # Prediction for hands that are splittable
            if(playerTotal % 2 == 0 and playerTotal >= 4 and playerTotal <= 20):
                splitState = (dealerCard, False, playerTotal, True)
                splitState = np.reshape(splitState, [1, observationSpace])

                if(name=='DQfD'):
                    dqPred, nstepPred, slmcPred = agent.trainableModel.predict([splitState, splitState, splitState])
                    QValues = dqPred + nstepPred + slmcPred
                elif(name=='DQN'):
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

                if(name=='DQfD'):
                    dqPred, nstepPred, slmcPred = agent.trainableModel.predict([aceState, aceState, aceState])
                    QValues = dqPred + nstepPred + slmcPred
                elif(name=='DQN'):
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

            # Prediction for regular hands (no useable ace and not splittable)
            normalState = (dealerCard, False, playerTotal, False)
            normalState = np.reshape(normalState, [1, observationSpace])

            if(name=='DQfD'):
                dqPred, nstepPred, slmcPred = agent.trainableModel.predict([normalState, normalState, normalState])
                QValues = dqPred + nstepPred + slmcPred
            elif(name=='DQN'):
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
        
        if(name=='DQfD'):
            dqPred, nstepPred, slmcPred = agent.trainableModel.predict([doubleAceState, doubleAceState, doubleAceState])
            QValues = dqPred + nstepPred + slmcPred
        elif(name=='DQN'):
            QValues = agent.model.predict(doubleAceState)
        
        t.add_row([12, 'True', 'True', dealerCard, QValues[0][0], QValues[0][1], QValues[0][2], QValues[0][3]])
        bestAction = getBestAction(QValues, True)
        twinAce[dealerCard - 1] = bestAction

    print(t)
    print("")
    fileName = 'Results_' + str(int(numGames/1000)) + 'K_Games.csv'
    createDF(fileName)
    return fileName