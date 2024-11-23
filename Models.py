from torch import nn
import torch


class FFN(nn.Module):
    def __init__(self, inputSize=18, outputSize=9, hiddenSizes=(32,), conversionSizes=(32, 16)):
        super().__init__()
        assert len(hiddenSizes) > 0
        self.inputLayer = nn.Linear(inputSize, hiddenSizes[0])
        self.hiddenLayers = nn.ModuleList()
        for hiddenIndex in range(1, len(hiddenSizes)):
            self.hiddenLayers.append(nn.Linear(hiddenSizes[hiddenIndex - 1], hiddenSizes[hiddenIndex]))

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
        for hiddenLayer in self.hiddenLayers:
            x = relu(hiddenLayer(x))
        outputs = []
        for layerList in self.conversionLayers:
            tempX = x
            for layer in layerList:
                tempX = relu(layer(tempX))
            outputs.append(tempX)

        x = torch.cat(outputs, dim=1)

        return x
