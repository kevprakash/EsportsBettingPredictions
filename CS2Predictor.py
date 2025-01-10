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

def linearPredictor():
    data = getMergedTable()
    xCols = ["kills_avg", "headshots_avg", "assists_avg", "deaths_avg", "Rounds_avg", "KPR_avg", "HSPR_avg", "APR_avg", "DPR_avg", "odds_avg", "odds"]
    yCols = ["kills", "headshots", "Rounds"]

    x = data[xCols].to_numpy()
    y = data[yCols].to_numpy()
    ypr = y[:, 0:2] / y[:, 2:3]
    y = np.concatenate([ypr, y[:, 2:3]], axis=1)

    # print(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    rmseValues = [mean_squared_error(y_test[:, i], y_pred[:, i]) ** 0.5 for i in range(y_test.shape[1])]
    r2Values = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]

    for i in range(y_test.shape[1]):
        print(str(rmseValues[i]) + "\t|\t" + str(r2Values[i]))


def FFNPredictor():
    data = getMergedTable()
    xCols = ["kills_avg", "headshots_avg", "assists_avg", "deaths_avg", "Rounds_avg", "KPR_avg", "HSPR_avg", "APR_avg", "DPR_avg", "odds_avg", "odds"]
    yCols = ["kills", "Rounds", "line_score"]

    x = data[xCols].to_numpy()
    y = data[yCols].to_numpy()
    yPredTarget = data["kills"].to_numpy()
    ypr = y[:, 0:1] / y[:, 1:2]
    y = np.concatenate([ypr, y[:, 1:2] / 52, yPredTarget[:, None], y[:, 2:3]], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = FFN()

    def train(xData, yData, epochs=100):
        model.cuda()
        optim = Adam(model.parameters(), lr=1e-4)

        lossFunc = MSELoss()
        batchSize = 4

        x = xData.copy()
        y = yData[:, 0:2].copy()
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

                predY = model(batch)
                # print(torch.cat([predY, target], dim=1))
                loss = lossFunc(predY, target)

                losses.append(loss.item() ** 0.5)
                if len(losses) % 5 == 0:
                    print(sum(losses[-5:]) / 5)
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
            predictions = model(torch.Tensor(xData))
            return predictions

    y_pred = predict(x_test)
    y_pred = y_pred[:, 0] * y_pred[:, 1] * 52

    '''
    rmseValues = [mean_squared_error(y_test[:, i], y_pred[:, i]) ** 0.5 for i in range(y_test.shape[1])]
    r2Values = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]

    for i in range(y_test.shape[1]):
        print(str(rmseValues[i]) + "\t|\t" + str(r2Values[i]))
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test[:, i], y_pred[:, i], color='blue', label=f'Target Variable {i + 1}')
        plt.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], color='red',
                 linestyle='--')  # Line of perfect prediction
        plt.xlabel(f'True Values (y{i + 1})')
        plt.ylabel(f'Predicted Values (y{i + 1})')
        plt.legend()
        plt.grid(True)
        plt.show()
    '''

    # print(np.stack([y_test[:, 2], y_pred.numpy()], axis=1))
    print(mean_squared_error(y_test[:, 2], y_pred) ** 0.5)
    print(r2_score(y_test[:, 2], y_pred))

    line = y_test[:, 3]
    actualKills = y_test[:, 2]
    predictedKills = y_pred.numpy()

    overActual = actualKills > line
    overPredict = predictedKills > line

    hits = overActual == overPredict
    hitCount = np.sum(hits)
    print(hitCount / len(hits) * 100)


FFNPredictor()