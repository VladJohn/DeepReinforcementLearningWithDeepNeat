import torch
import copy

def randomUniformValue():
    return float(torch.rand(1))


def randomBoolean():
    return randomUniformValue() <= 0.5


def getBestPerformingNetwork(population):
    populationCopy = copy.deepcopy(population)
    populationCopy.sort(key=lambda network: network.fitness, reverse=True)
    return populationCopy[0]