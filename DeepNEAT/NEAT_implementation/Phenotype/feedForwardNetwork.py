import torch
import torch.nn as nn
import torch.nn.functional as F
import DeepNEAT.NEAT_implementation.Activations.activations as activations
from torch import autograd
import random

DEVICE = torch.device("cpu")


class FeedForwardNetwork(nn.Module):

    def __init__(self, genome, config):
        super(FeedForwardNetwork, self).__init__()
        self.genome = genome
        self.config = config
        self.values = self.buildValues(config)
        self.nn_modules = nn.ModuleList()

        for value in self.values:
            self.nn_modules.append(value.node)

    def forward(self, x):
        stackedValues = self.genome.orderNodesByValue(self.values)

        while len(stackedValues) > 0:
            currentValue = stackedValues.pop()

            if (currentValue.referenceNode.type == 'input'):
                x = F.relu(currentValue.node(x))

            if (currentValue.referenceNode.type == 'output'):
                x = x.view(-1, currentValue.referenceNode.inputs)
                x = currentValue.node(x)

            if currentValue.referenceNode.type != 'input' and currentValue.referenceNode.type != 'output':
                x = x.view(-1, currentValue.referenceNode.inputs)
                if currentValue.referenceNode.type == 'conv1d':
                    if currentValue.referenceNode.activation == 'relu':
                        x = F.max_pool1d(F.relu(currentValue.node(x)), 2)
                    if currentValue.referenceNode.activation == 'sigmoid':
                        x = F.max_pool1d(F.sigmoid(currentValue.node(x)), 2)
                    if currentValue.referenceNode.activation == 'tanh':
                        x = F.max_pool1d(F.tanh(currentValue.node(x)), 2)
                else:
                    if currentValue.referenceNode.activation == 'relu':
                        x = F.relu(currentValue.node(x))
                    if currentValue.referenceNode.activation == 'sigmoid':
                        x = F.sigmoid(currentValue.node(x))
                    if currentValue.referenceNode.activation == 'tanh':
                        x = F.tanh(currentValue.node(x))
        return F.softmax(x, 1)

    def buildValues(self, config):
        values = []

        for node in self.genome.nodes:
            inputConnections = self.genome.getInputConnections(node.id)

            newValue = Value(node, config)

            values.append(newValue)
        return values


class Value:

    def __init__(self, referenceNode, config):
        self.referenceNode = referenceNode
        if self.referenceNode.type == 'input':
            self.node = self.buildLinear(self.referenceNode.inputs, self.referenceNode.outputs)
        if self.referenceNode.type == 'output':
            self.node = self.buildLinear(self.referenceNode.inputs, self.referenceNode.outputs)
        if self.referenceNode.type == 'linear':
            self.node = self.buildLinear(self.referenceNode.inputs, self.referenceNode.outputs)
        if self.referenceNode.type == 'conv1d':
            self.node = self.buildConv1d(self.referenceNode.inputs, self.referenceNode.outputs, self.referenceNode.kernelSize, self.referenceNode.stride)
        self.activation = self.referenceNode.activation

    def buildLinear(self, inputs, outputs):
        return nn.Linear(inputs, outputs, False)

    def buildConv1d(self, inputs, outputs, kernelSize, stride):
        return nn.Conv1d(inputs, outputs, kernelSize, stride)