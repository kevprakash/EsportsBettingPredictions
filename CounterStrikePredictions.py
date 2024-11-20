import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from Models import FFN
from torch.nn import MSELoss
from torch.optim import Adam
import torch

pd.set_option('display.max_columns', None)


def readAndProcessRawData(load=False, perRound=False):
    IDCols = ["Name", "Date", "Map", "Map Number"]
    totalValues = ["Kills", "Headshots", "Assists", "Deaths"]
    summaryValues = ["Kast", "ADR", "Rating"]
    valueCols = totalValues + summaryValues

    if load:
        fittingData = pd.read_csv("Data/CS/RawFittingData.csv" if perRound else "Data/CS/RawFittingDataTotals.csv")
    else:
        data2020 = pd.read_csv("Data/CS/2020.csv", index_col=0)
        data2020["Date"] = pd.to_datetime(data2020["Date"], format='%m/%d/%Y %H:%M')
        data2023 = pd.read_csv("Data/CS/2023.csv", index_col=0)
        data2023["Date"] = pd.to_datetime(data2023["Date"], format='%Y-%m-%d %H:%M')
        data2024 = pd.read_csv("Data/CS/2024.csv", index_col=0)
        data2024["Date"] = pd.to_datetime(data2024["Date"], format='%Y-%m-%d %H:%M')

        data = pd.concat([data2020, data2023, data2024])

        data["Kast"] = data["Kast"].replace('%', '', regex=True).astype("float")
        data["ADR"] = data["ADR"].replace("-", 0).astype("float")

        truncatedData = data[IDCols + valueCols + ["Team Score", "Opponent Score"]].copy()

        if perRound:
            # Change relevant metrics to per-round
            truncatedData['Rounds'] = truncatedData["Team Score"] + truncatedData["Opponent Score"]
            truncatedData = truncatedData[truncatedData['Rounds'] >= 13]  # Get rid of rows with bad round counts
            for tv in totalValues:
                truncatedData[tv] = truncatedData[tv] / truncatedData["Rounds"]

        fittingData = truncatedData.dropna().copy()
        fittingData = fittingData.sort_values(by=IDCols)

    return fittingData


def readAndProcessWMAData(load=False, perRound=False):
    IDCols = ["Name", "Date", "Map", "Map Number"]
    totalValues = ["Kills", "Headshots", "Assists", "Deaths"]
    summaryValues = ["Kast", "ADR", "Rating"]
    valueCols = totalValues + summaryValues

    if load:
        fittingData = pd.read_csv("Data/CS/FittingData.csv" if perRound else "Data/CS/FittingDataTotals.csv")
    else:
        data2020 = pd.read_csv("Data/CS/2020.csv", index_col=0)
        data2020["Date"] = pd.to_datetime(data2020["Date"], format='%m/%d/%Y %H:%M')
        data2023 = pd.read_csv("Data/CS/2023.csv", index_col=0)
        data2023["Date"] = pd.to_datetime(data2023["Date"], format='%Y-%m-%d %H:%M')
        data2024 = pd.read_csv("Data/CS/2024.csv", index_col=0)
        data2024["Date"] = pd.to_datetime(data2024["Date"], format='%Y-%m-%d %H:%M')

        data = pd.concat([data2020, data2023, data2024])

        data["Kast"] = data["Kast"].replace('%', '', regex=True).astype("float")
        data["ADR"] = data["ADR"].replace("-", 0).astype("float")

        truncatedData = data[IDCols + valueCols + ["Team Score", "Opponent Score"]].copy()

        if perRound:
            # Change relevant metrics to per-round
            truncatedData['Rounds'] = truncatedData["Team Score"] + truncatedData["Opponent Score"]
            truncatedData = truncatedData[truncatedData['Rounds'] >= 13]    # Get rid of rows with bad round counts
            for tv in totalValues:
                truncatedData[tv] = truncatedData[tv]/truncatedData["Rounds"]

        playerAvg = truncatedData.sort_values(by=IDCols)

        windowLength = 8

        weights = [0.5 ** i for i in range(1, windowLength)]
        weights = np.array(weights + [1 - sum(weights)])

        def truncateWeight(w, wl):
            tw = w[:wl]
            tw = tw[::-1]
            tw = tw/sum(tw)
            return tw

        adjustedWeights = [truncateWeight(weights, l) for l in range(len(weights) + 1)]

        def weightedMovingAverage(x):
            truncatedWeights = adjustedWeights[len(x)]
            return np.dot(x, truncatedWeights)

        def applyWMA(g, prefix=''):
            for col in valueCols:
                p = (prefix + ' ') if len(prefix) > 0 else ''
                g[p + col + " Avg"] = g[col].rolling(window=len(weights), min_periods=1, closed="left").apply(weightedMovingAverage, raw=True)
            return g

        def applyWMAMap(g):
            return applyWMA(g, prefix='Map')

        def applyWMAMapNum(g):
            return applyWMA(g, prefix='Map Num')

        playerAvg = playerAvg.groupby("Name").apply(applyWMA).reset_index(drop=True)
        playerAvg = playerAvg.groupby(["Name", "Map"]).apply(applyWMAMap).reset_index(drop=True)
        playerAvg = playerAvg.groupby(["Name", "Map Number"]).apply(applyWMAMapNum).reset_index(drop=True)

        fittingData = playerAvg.dropna().copy()

        fittingData.to_csv("Data/CS/FittingData.csv" if perRound else "Data/CS/FittingDataTotals.csv")

    xCols = [[x + " Avg", "Map " + x + " Avg", "Map Num " + x + " Avg"] for x in valueCols]
    xCols = [x for l in xCols for x in l]
    # xCols = [x + " Avg" for x in valueCols]

    x = fittingData[xCols + ["Team Score", "Opponent Score"]].to_numpy()

    yK = fittingData[["Kills"]].to_numpy()
    yH = fittingData[["Headshots"]].to_numpy()

    return fittingData, x, yK, yH, xCols


# lrK = LinearRegression(fit_intercept=False)
# lrH = LinearRegression(fit_intercept=False)
networkModel = FFN(hiddenSizes=[64, 64, 32])


def train(xData, yKills, yHeadshots, epochs=5):
    optim = Adam(networkModel.parameters(), lr=1e-5)    # 1e-5 best LR so far

    def MAPE(yPred, yReal, eps=1e-5):
        APE = torch.abs((yReal - yPred)/(yReal + eps))
        return torch.mean(APE)

    lossFunc = MSELoss()
    batchSize = 64                                      # 64 best so far

    print(xData.shape)

    x = xData
    y = np.concatenate([yKills, yHeadshots], axis=1)
    losses = []

    for e in range(epochs):
        startIndex = 0
        endIndex = min(len(xData), startIndex + batchSize)

        xy = np.concatenate([x, y], axis=1)
        np.random.shuffle(xy)

        x = torch.Tensor(xy[:, :-2])
        y = torch.Tensor(xy[:, -2:])

        while startIndex < len(x):
            batch = x[startIndex:endIndex]
            target = y[startIndex:endIndex]

            predY = networkModel(batch)
            # print(torch.cat([predY, target], dim=1))
            loss = lossFunc(predY, target) ** 0.5

            losses.append([loss.item()])
            if len(losses) % 100 == 0:
                print(loss.item())
            loss.backward()
            optim.step()

            startIndex = endIndex
            endIndex = min(len(x), endIndex + batchSize)

    return losses


def predict(xData):
    with torch.no_grad():
        predictions = networkModel(torch.Tensor(xData))
        return predictions[:, 0].numpy(), predictions[:, 1].numpy()
    # return lrK.predict(xData), lrH.predict(xData)


def run():
    fittingData, x, yK, yH, xCols = readAndProcessWMAData(load=True)
    trainSize = int(len(x) * 0.9)
    train(x[:trainSize], yK[:trainSize], yH[:trainSize])
    fittingData["Kills Pred"], fittingData["Headshots Pred"] = predict(x)

    check = fittingData[["Name", "Date", "Map", "Map Number", "Kills Pred", "Kills", "Headshots Pred", "Headshots"] + xCols].copy()
    check = check[trainSize:]
    check["Kills Error"] = check["Kills Pred"] - check["Kills"]
    check["Headshots Error"] = check["Headshots Pred"] - check["Headshots"]

    def summarize(df):
        TSS_K = np.sum((df["Kills"] - df["Kills"].mean()) ** 2)
        RSS_K = np.sum((df["Kills Error"]) ** 2)
        R2_K = 1 - RSS_K / TSS_K

        TSS_H = np.sum((df["Headshots"] - df["Headshots"].mean()) ** 2)
        RSS_H = np.sum((df["Headshots Error"]) ** 2)
        R2_H = 1 - RSS_H / TSS_H

        print("R^2 Kills: " + str(R2_K))
        print("R^2 HS: " + str(R2_H))
        print("Kills MAE: " + str(np.abs(df["Kills Error"]).mean()))
        print("HS MAE: " + str(np.abs(df["Headshots Error"]).mean()))
        print("Kills RMSE: " + str(np.sqrt((df["Kills Error"] ** 2).mean())))
        print("HS RMSE: " + str(np.sqrt((df["Headshots Error"] ** 2).mean())))

    print()
    print("Individual Maps")
    summarize(check)

    print()
    aggregate = check.groupby(["Name", "Date"]).sum()

    print("Match Totals")
    summarize(aggregate)

    def displayPredictions(df, category='Kills'):
        plt.scatter(df[category], df[category + " Pred"], color="blue", label='Data Points')
        plt.plot(df[category], df[category], color='red', linestyle='--', label='x = y')

        plt.xlabel(category)
        plt.ylabel("Predicted " + category)
        plt.legend()
        plt.grid(True)

        plt.show()

    displayPredictions(check)
    displayPredictions(check, category='Headshots')
    displayPredictions(aggregate)
    displayPredictions(aggregate, category='Headshots')


run()
