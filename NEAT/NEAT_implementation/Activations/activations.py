import torch as torch


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


class Activations:

    def __init__(self):
        self.functions = dict(
            sigmoid=sigmoid,
            tanh=tanh
        )

    def get(self, functionName):
        return self.functions.get(functionName, None)