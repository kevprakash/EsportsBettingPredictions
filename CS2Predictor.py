import pandas as pd
import numpy as np
from TableReader import getMergedTable
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Models import FFN
import torch
from torch.optim import Adam
from torch.nn import MSELoss
import matplotlib.pyplot as plt


def getXY():
    data = getMergedTable()
    # xCols = ["kills_avg", "headshots_avg", "HS%_avg", "assists_avg", "deaths_avg", "Rounds_avg", "KPR_avg", "HSPR_avg", "APR_avg", "DPR_avg", "odds_avg", "odds"]
    xCols = ["HS%_avg", "Round%_avg", "KPR_avg", "odds_avg", "odds"]
    yCols = ["kills", "headshots"]
    # yCols = ["line_score", "line_score_HS"]

    x = data[xCols].to_numpy()
    yTargets = data[yCols].to_numpy()
    yRounds = data[["Round%"]].to_numpy()
    yLines = data[["line_score", "line_score_HS"]].to_numpy()
    # yKPR = data[["kills"]].to_numpy() / yRounds
    yKPR = data[["line_score"]].to_numpy() / data[["Rounds"]].to_numpy()
    # yHSP = data[["HS%"]].to_numpy()
    yHSP = data[["line_score_HS"]].to_numpy() / data[["line_score"]].to_numpy()

    y = np.concatenate([yKPR, yHSP, yRounds, yTargets, yLines], axis=1)

    return x, y


def linearPredictor():

    x, y = getXY()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(x_train, y_train[:, :3])

    y_rawPred = model.predict(x_test)
    yRounds = y_rawPred[:, 2] * 26 + 26
    yKills = y_rawPred[:, 0] * yRounds
    yHS = y_rawPred[:, 1] * yKills
    y_pred = np.stack([yRounds, yKills, yHS], axis=1)

    y_targets = np.concatenate([y_test[:, 2:3] * 26 + 26, y_test[:, -4:-2]], axis=1)
    lines = y_test[:, -2:]

    evaluate(y_targets, y_pred, lines)


def FFNPredictor():
    x, y = getXY()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = FFN(inputSize=x.shape[1], outputSize=3)

    def train(xData, yData, epochs=100):
        model.cuda()
        optim = Adam(model.parameters(), lr=10**-4)

        lossFunc = MSELoss()
        batchSize = 2

        x = xData.copy()
        y = yData[:, :3].copy()
        losses = []

        for e in range(epochs):
            startIndex = 0
            endIndex = min(len(xData), startIndex + batchSize)

            xSize = x.shape[1]

            xy = np.concatenate([x, y], axis=1)
            np.random.shuffle(xy)

            xTensor = torch.Tensor(xy[:, :xSize]).cuda()
            yTensor = torch.Tensor(xy[:, xSize:]).cuda()
            batchCount = 0

            while startIndex < len(x):
                batch = xTensor[startIndex:endIndex]
                target = yTensor[startIndex:endIndex]

                tempY = model(batch)
                # yR = tempY[:, 2]
                # yKills = tempY[:, 0]
                # yHS = yKills * tempY[:, 1]

                predY = tempY

                # print(torch.cat([predY, target], dim=1))
                loss = lossFunc(predY, target)

                losses.append(loss.item() ** 0.5)
                # if len(losses) % 2 == 0:
                #     print(sum(losses[-2:]) / 2)
                loss.backward()
                optim.step()

                startIndex = endIndex
                endIndex = min(len(x), endIndex + batchSize)
                batchCount += 1

            print("Epoch " + str(e + 1) + " Avg Loss:", sum(losses[-batchCount:]) / batchCount)

        model.cpu()
        print()

        return losses

    train(x_train, y_train)

    def predict(xData):
        with torch.no_grad():
            tempY = model(torch.Tensor(xData))

            yR = tempY[:, 2]
            yKills = tempY[:, 0] * (yR * 26 + 26)
            yHS = yKills * tempY[:, 1]

            predictions = torch.stack([yR, yKills, yHS], dim=-1)

            return predictions

    y_pred = predict(x_test).numpy()

    # print(np.stack([y_test[:, 2], y_pred.numpy()], axis=1))
    targets = y_test[:, -5:-2]

    lines = y_test[:, -2:]

    evaluate(targets, y_pred, lines)


def evaluate(target, pred, lines):
    rmseValues = [mean_squared_error(target[:, i], pred[:, i]) ** 0.5 for i in range(target.shape[1])]
    r2Values = [r2_score(target[:, i], pred[:, i]) for i in range(target.shape[1])]

    for i in range(target.shape[1]):
        # print(str(rmseValues[i]) + "\t|\t" + str(r2Values[i]))

        plt.figure(figsize=(6, 4))
        plt.scatter(target[:, i], pred[:, i], color='blue', label=f'Target Variable {i + 1}')
        plt.plot([min(target[:, i]), max(target[:, i])], [min(target[:, i]), max(target[:, i])],
                 color='red',
                 linestyle='--')  # Line of perfect prediction
        plt.xlabel(f'True Values (y{i + 1})')
        plt.ylabel(f'Predicted Values (y{i + 1})')
        plt.legend()
        plt.grid(True)
        plt.show()

    actualVals = target[:, -2:]
    predictedVals = pred[:, -2:]

    overActual = actualVals >= lines
    overPredict = predictedVals >= lines

    underActual = actualVals < lines
    underPredict = predictedVals < lines

    inversePrediction = overActual == underPredict

    overKillsHit = overPredict[:, 0][overActual[:, 0]]
    underKillsHit = underPredict[:, 0][underActual[:, 0]]
    overHSHit = overPredict[:, 1][overActual[:, 1]]
    underHSHit = underPredict[:, 1][underActual[:, 1]]

    overCount = np.sum(overActual, axis=0)
    underCount = np.sum(underActual, axis=0)

    overHitCount = np.stack([np.sum(overKillsHit), np.sum(overHSHit)], axis=-1)
    underHitCount = np.stack([np.sum(underKillsHit), np.sum(underHSHit)], axis=-1)

    n = len(target)

    print("Over Hits: " + str(overHitCount/overCount) + " @ " + str(overCount / n))
    print("Under Hits: " + str(underHitCount / underCount) + " @ " + str(underCount / n))
    print()
    print("Total Accuracy: " + str((overHitCount + underHitCount) / n))
    print("Inverse Accuracy: " + str(np.sum(inversePrediction, axis=0) / n))
    print()
    print("Over % of Guesses: " + str(np.sum(overPredict, axis=0) / n))


linearPredictor()
# FFNPredictor()