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
    IDCols = ["Name", "Date", "Map Number", "Map"]
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
        data["Map Number"] = data["Map Number"].replace("Single Map", 1).astype("float")

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


windowLength = 8
weights = [0.5 ** i for i in range(1, windowLength)]
weights = np.array(weights + [1 - sum(weights)])


def truncateWeight(w, wl):
    tw = w[:wl]
    tw = tw[::-1]
    tw = tw / sum(tw)
    return tw


def weightedMovingAverage(x):

    adjustedWeights = [truncateWeight(weights, l) for l in range(len(weights) + 1)]
    truncatedWeights = adjustedWeights[len(x)]
    return np.dot(x, truncatedWeights)


def readAndProcessWMAData(load=False, perRound=False):
    IDCols = ["Name", "Date", "Map", "Map Number"]
    totalValues = ["Kills", "Headshots", "Assists", "Deaths"]
    summaryValues = ["Kast", "ADR", "Rating", "Team Score", "Opponent Score"]
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

        truncatedData = data[IDCols + valueCols].copy()

        if perRound:
            # Change relevant metrics to per-round
            truncatedData['Rounds'] = truncatedData["Team Score"] + truncatedData["Opponent Score"]
            truncatedData = truncatedData[truncatedData['Rounds'] >= 13]    # Get rid of rows with bad round counts
            for tv in totalValues:
                truncatedData[tv] = truncatedData[tv]/truncatedData["Rounds"]

        playerAvg = truncatedData.sort_values(by=IDCols)

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
        # playerAvg = playerAvg.groupby(["Name", "Map"]).apply(applyWMAMap).reset_index(drop=True)
        playerAvg = playerAvg.groupby(["Name", "Map Number"]).apply(applyWMAMapNum).reset_index(drop=True)

        fittingData = playerAvg.dropna().copy()

        fittingData.to_csv("Data/CS/FittingData.csv" if perRound else "Data/CS/FittingDataTotals.csv")

    # xCols = [[x + " Avg", "Map " + x + " Avg", "Map Num " + x + " Avg"] for x in valueCols]
    xCols = [[x + " Avg", "Map Num " + x + " Avg"] for x in valueCols]
    # xCols = [[x + " Avg"] for x in valueCols]
    xCols = [x for l in xCols for x in l]
    # xCols = [x + " Avg" for x in valueCols]

    print(xCols)

    x = fittingData[xCols].to_numpy()

    y = fittingData[valueCols].to_numpy()

    return fittingData, x, y, xCols


# lrK = LinearRegression(fit_intercept=False)
# lrH = LinearRegression(fit_intercept=False)
networkModel = FFN(hiddenSizes=[128, 64])


def loadModel():
    networkModel.load_state_dict(torch.load("Models/CSPredModel.pt", weights_only=True))
    networkModel.eval()


def train(xData, yData, epochs=5):
    networkModel.cuda()
    optim = Adam(networkModel.parameters(), lr=1e-5)    # 1e-5 best LR so far

    def BalancedMSE(yPred, yReal):
        assert yPred.shape == yReal.shape
        mse = MSELoss(reduction='none')(yPred, yReal)
        errorMeans = torch.mean(mse, dim=0)
        realMeans = torch.mean(yReal, dim=0)
        weightedMean = errorMeans/(realMeans ** 2)
        return torch.mean(weightedMean, dim=0)

    lossFunc = BalancedMSE
    batchSize = 64                                      # 64 best so far

    print(xData.shape)
    print(yData.shape)

    x = xData.copy()
    y = yData.copy()
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

            predY = networkModel(batch)
            # print(torch.cat([predY, target], dim=1))
            loss = lossFunc(predY, target)

            losses.append(loss.item() ** 0.5)
            if len(losses) % 1000 == 0:
                print(sum(losses[-1000:])/1000)
            loss.backward()
            optim.step()

            startIndex = endIndex
            endIndex = min(len(x), endIndex + batchSize)
            batchCount += 1

        print("Epoch " + str(e + 1) + " Avg Loss:", sum(losses[-batchCount:])/batchCount)

    networkModel.cpu()
    torch.save(networkModel.state_dict(), "Models/CSPredModel.pt")

    return losses


def predict(xData):
    with torch.no_grad():
        predictions = networkModel(torch.Tensor(xData))
        return predictions[:, 0].numpy(), predictions[:, 1].numpy()
    # return lrK.predict(xData), lrH.predict(xData)


def run():
    fittingData, x, y, xCols = readAndProcessWMAData(load=True)
    trainSize = int(len(x) * 0.9)
    train(x[:trainSize], y[:trainSize])
    # loadModel()
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


def getLatestPlayerData(load=False):
    IDCols = ["Name", "Date", "Map Number", "Map"]
    totalValues = ["Kills", "Headshots", "Assists", "Deaths"]
    summaryValues = ["Kast", "ADR", "Rating"]
    valueCols = totalValues + summaryValues  + ["Team Score", "Opponent Score"]

    if not load:
        data = pd.read_csv("Data/CS/2024.csv")
        data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d %H:%M')
        data["Kast"] = data["Kast"].replace('%', '', regex=True).astype("float")
        data["ADR"] = data["ADR"].replace("-", 0).astype("float")
        data["Map Number"] = data["Map Number"].replace("Single Map", 1).astype("float")

        truncatedData = data[IDCols + valueCols].copy()
        playerAvg = truncatedData.dropna().copy().sort_values(by=IDCols)

        def applyWMA(g, prefix=''):
            for col in valueCols:
                p = (prefix + ' ') if len(prefix) > 0 else ''
                g[p + col + " Avg"] = g[col].rolling(window=len(weights), min_periods=1).apply(weightedMovingAverage, raw=True)
            return g

        def applyWMAMap(g):
            return applyWMA(g, prefix='Map')

        def applyWMAMapNum(g):
            return applyWMA(g, prefix='Map Num')

        playerAvg = playerAvg.groupby("Name").apply(applyWMA).reset_index(drop=True)
        # playerAvg = playerAvg.groupby(["Name", "Map"]).apply(applyWMAMap).reset_index(drop=True)
        playerAvg = playerAvg.groupby(["Name", "Map Number"]).apply(applyWMAMapNum).reset_index(drop=True)

        latestData = playerAvg.groupby(["Name"]).tail(1)

        print(latestData)

        predData = playerAvg.dropna().copy()

        predData.to_csv("Data/CS/PredictionData.csv")

    else:
        predData = pd.read_csv("Data/CS/PredictionData.csv")

    return predData


def PrizePicksComparison():
    fittingData, x, y, xCols = readAndProcessWMAData(load=True)
    loadModel()
    networkModel.eval()
    predData = getLatestPlayerData(load=True)
    latestData = predData.groupby("Name").tail(1).copy()
    map1Data = predData[predData["Map Number"] == 1.0].groupby("Name").tail(1).copy().reset_index()
    map2Data = predData[predData["Map Number"] == 2.0].groupby("Name").tail(1).copy().reset_index()
    map1Data = map1Data[map1Data["Name"].isin(map2Data["Name"])].reset_index()
    map2Data = map2Data[map2Data["Name"].isin(map1Data["Name"])].reset_index()
    map3Data = predData[predData["Map Number"] == 3.0].groupby("Name").tail(1).copy().reset_index()

    map1X = latestData[latestData["Name"].isin(map1Data["Name"])].copy().reset_index()
    map1X[xCols[1::2]] = map1Data[xCols[1::2]]
    map1X = torch.Tensor(map1X[xCols].to_numpy())

    with torch.no_grad():
        map1Preds = networkModel(map1X).numpy()

    pred1 = map1Data[["Name"]].copy()
    pred1["Kills 1 Pred"] = map1Preds[:, 0]
    pred1["HS 1 Pred"] = map1Preds[:, 1]

    map2X = latestData[latestData["Name"].isin(map2Data["Name"])].copy().reset_index()
    map2X[xCols[::2]] = (map2X[xCols[::2]] + map1Preds)/2
    map2X[xCols[1::2]] = map2Data[xCols[1::2]]
    map2X = torch.Tensor(map2X[xCols].to_numpy())

    with torch.no_grad():
        map2Preds = networkModel(map2X).numpy()

    pred2 = map2Data[["Name"]].copy()
    pred2["Kills 2 Pred"] = map2Preds[:, 0]
    pred2["HS 2 Pred"] = map2Preds[:, 1]

    pred12 = pd.merge(pred1, pred2, on="Name", how="inner")
    pred12["Kills 1-2 Pred"] = pred12["Kills 1 Pred"] + pred12["Kills 2 Pred"]
    pred12["HS 1-2 Pred"] = pred12["HS 1 Pred"] + pred12["HS 2 Pred"]

    map3X = latestData[latestData["Name"].isin(map3Data["Name"])].copy().reset_index()
    filteredPred2 = pred2[["Name"]].copy().reset_index()
    filteredPred2[xCols[::2]] = map2Preds
    map3X[xCols[::2]] = (map3X[xCols[::2]] + filteredPred2[xCols[::2]]) / 2
    map3X[xCols[1::2]] = map3Data[xCols[1::2]]
    map3X = torch.Tensor(map3X[xCols].to_numpy())

    with torch.no_grad():
        map3Preds = networkModel(map3X).numpy()

    pred3 = map3Data[["Name"]].copy()
    pred3["Kills 3 Pred"] = map3Preds[:, 0]
    pred3["HS 3 Pred"] = map3Preds[:, 1]

    relevantCols = ["Stat Type", "Player Name", "Line", "Actual"]
    ppData = pd.read_csv("Data/CS/1day_hit_rate.csv")[relevantCols]
    ppData = ppData.rename(columns={"Player Name": "Name"})
    kills12Data = ppData[ppData["Stat Type"] == "MAPS 1-2 Kills"]
    hs12Data = ppData[ppData["Stat Type"] == "MAPS 1-2 Headshots"]
    kills3Data = ppData[ppData["Stat Type"] == "MAPS 3 Kills"]
    hs3Data = ppData[ppData["Stat Type"] == "MAPS 3 Headshots"]

    kills12Data = pd.merge(kills12Data, pred12[["Name", "Kills 1-2 Pred"]], on="Name", how="inner")
    kills12Data = kills12Data.rename(columns={"Kills 1-2 Pred": "Prediction"})
    hs12Data = pd.merge(hs12Data, pred12[["Name", "HS 1-2 Pred"]], on="Name", how="inner")
    hs12Data = hs12Data.rename(columns={"HS 1-2 Pred": "Prediction"})

    kills3Data = pd.merge(kills3Data, pred3[["Name", "Kills 3 Pred"]], on="Name", how="inner")
    kills3Data = kills3Data.rename(columns={"Kills 3 Pred": "Prediction"})
    hs3Data = pd.merge(hs3Data, pred3[["Name", "HS 3 Pred"]], on="Name", how="inner")
    hs3Data = hs3Data.rename(columns={"HS 3 Pred": "Prediction"})

    merged12 = pd.concat([kills12Data, hs12Data], ignore_index=True)
    merged3 = pd.concat([kills3Data, hs3Data], ignore_index=True)

    merged = pd.concat([merged12, merged3], ignore_index=True)
    merged = merged.dropna()

    def evaluate(df, label):
        df["Over"] = df["Actual"] > df["Line"]
        df["Predicted Over"] = df["Prediction"] > df["Line"]
        hits = (df["Over"] == df["Predicted Over"]).sum()
        acc = hits / len(df)

        print(label + ": " + f"{acc * 100:.2f}%")

    print()
    evaluate(merged12, "Map 1-2")
    evaluate(merged3, "Map 3")
    evaluate(merged, "Overall")


# run()
PrizePicksComparison()
