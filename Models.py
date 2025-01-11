from torch import nn
import torch


class FFN(nn.Module):
    def __init__(self, inputSize=12, outputSize=3, hiddenSizes=(32, 32, 32), conversionSizes=(16, 4)):
        super().__init__()
        assert len(hiddenSizes) > 0
        self.inputLayer = nn.Linear(inputSize, hiddenSizes[0])
        self.hiddenLayers = nn.ModuleList()
        self.batchNorms = nn.ModuleList()
        for hiddenIndex in range(1, len(hiddenSizes)):
            outSize = hiddenSizes[hiddenIndex]
            self.hiddenLayers.append(nn.Linear(hiddenSizes[hiddenIndex - 1], outSize))
            self.batchNorms.append(nn.BatchNorm1d(outSize))
        self.conversionLayers = nn.ModuleList()
        for i in range(outputSize):
            layerList = nn.ModuleList()
            layerList.append(nn.Linear(hiddenSizes[-1], conversionSizes[0]))
            self.conversionLayers.append(layerList)
        for i in range(1, len(conversionSizes)):
            for layerList in self.conversionLayers:
                layerList.append(nn.Linear(conversionSizes[i-1], conversionSizes[i]))

        for layerList in self.conversionLayers:
            layerList.append(nn.Linear(conversionSizes[-1], 1))

    def forward(self, x):
        relu = nn.LeakyReLU()
        x = relu(self.inputLayer(x))
        for i in range(len(self.hiddenLayers)):
            layer = self.hiddenLayers[i]
            BN = self.batchNorms[i]
            x = relu(BN(layer(x)))
        outputs = []
        for layerList in self.conversionLayers:
            tempX = x
            for layer in layerList:
                tempX = relu(layer(tempX))
            outputs.append(tempX)

        x = torch.cat(outputs, dim=1)

        return x