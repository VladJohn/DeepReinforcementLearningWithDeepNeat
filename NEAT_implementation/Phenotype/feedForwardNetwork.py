import torch
import torch.nn as nn
import NEAT_implementation.Activations.activations as activations
from torch import autograd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeedForwardNetwork(nn.Module):

    def __init__(self, genome, config):
        super(FeedForwardNetwork, self).__init__()
        self.genome = genome
        self.values = self.buildValues()
        self.linearModules = nn.ModuleList()
        self.config = config
        self.activation = activations.Activations().get(config.ACTIVATION)

        for value in self.values:
            self.linearModules.append(value.linear)

    def forward(self, x):
        outputs = dict()
        inputValues = [value for value in self.values if value.referenceNode.type == 'input']
        outputValues = [value for value in self.values if value.referenceNode.type == 'output']
        biasValues = [value for value in self.values if value.referenceNode.type == 'bias']
        stackedValues = self.genome.orderNodesByValue(self.values)

        for value in inputValues:
            outputs[value.referenceNode.id] = x[0][value.referenceNode.id]

        for value in biasValues:
            outputs[value.referenceNode.id] = torch.ones((1, 1)).to(DEVICE)[0][0]

        while len(stackedValues) > 0:
            currentValue = stackedValues.pop()

            if currentValue.referenceNode.type != 'input' and currentValue.referenceNode.type != 'bias':

                nodeInputIds = self.genome.getNodeInputIds(currentValue.referenceNode.id)
                inputVector = autograd.Variable(torch.zeros((1, len(nodeInputIds)), device=DEVICE, requires_grad=True))

                for i, nodeInputId in enumerate(nodeInputIds):
                    inputVector[0][i] = outputs[nodeInputId]

                linearModule = self.linearModules[self.values.index(currentValue)]
                if linearModule is not None:
                    scaledActivation = self.config.SCALE_ACTIVATION * linearModule(inputVector)
                    output = self.activation(scaledActivation)
                else:
                    output = torch.zeros((1, 1))

                outputs[currentValue.referenceNode.id] = output

        outputVector = autograd.Variable(torch.zeros((1, len(outputValues)), device=DEVICE, requires_grad=True))
        for i, value in enumerate(outputValues):
            outputVector[0][i] = outputs[value.referenceNode.id]
        return outputVector

    def buildValues(self):
        values = []

        for node in self.genome.nodes:
            inputConnections = self.genome.getInputConnections(node.id)
            countInputConnections = len(inputConnections)
            weights = [connection.weight for connection in inputConnections]

            newValue = Value(node, countInputConnections)
            newValue.setWeights(weights)

            values.append(newValue)
        return values


class Value:

    def __init__(self, referenceNode, countInputConnections):
        self.referenceNode = referenceNode
        self.linear = self.buildLinear(countInputConnections)

    def setWeights(self, weights):
        if self.referenceNode.type != 'input' and self.referenceNode.type != 'bias':
            weights = torch.cat(weights).unsqueeze(0)
            for parameter in self.linear.parameters():
                parameter.data = weights

    def buildLinear(self, countInputConnections):
        if self.referenceNode.type == 'input' or self.referenceNode.type == 'bias':
            return None
        return nn.Linear(countInputConnections, 1, False)