class Node:
    def __init__(self, nodeId, nodeType):
        self.id = nodeId
        self.type = nodeType
        self.value = None
        self.inputs = None
        self.outputs = None
        self.activation = None
        self.kernelSize = None
        self.stride = None