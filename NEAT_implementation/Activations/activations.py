import torch.nn.functional as Functional


def sigmoid(x):
    return Functional.sigmoid(x)


def tanh(x):
    return Functional.tanh(x)


class Activations:

    def __init__(self):
        self.functions = dict(
            sigmoid=sigmoid,
            tanh=tanh
        )

    def get(self, functionName):
        return self.functions.get(functionName, None)