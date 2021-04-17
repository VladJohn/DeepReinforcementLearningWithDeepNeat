import random
from DeepNEAT.NEAT_implementation.Genotype.connection import Connection
from DeepNEAT.NEAT_implementation.Genotype.node import Node
import DeepNEAT.NEAT_implementation.Utils.utils as utils


class Genome:

    def __init__(self, configuration):
        self.connections = []
        self.nodes = []
        self.nodeIds = set()
        self.innovationNumbers = set()
        self.fitness = None
        self.adjustedFitness = None
        self.species = None
        self.configuration = configuration

    # def addConnectionMutation(self):
    #
    #     possibleInputs = [n.id for n in self.nodes if n.type != 'output']
    #     possibleOutputs = [n.id for n in self.nodes if n.type != 'input' and n.type != 'bias']
    #
    #     if len(possibleOutputs) is not 0 and len(possibleInputs) is not 0:
    #         inputNodeId = random.choice(possibleInputs)
    #         outputNodeId = random.choice(possibleOutputs)
    #
    #         if self.checkValidConnection(inputNodeId, outputNodeId):
    #             self.addConnection(inputNodeId, outputNodeId)

    def addNodeMutation(self, config):
        node_type = utils.randomUniformValue()
        if node_type > 0.5:
            newNode = self.addNode('conv1d')
            newNode.kernelSize = random.choice(config.KERNEL_SIZE)
            newNode.stride = random.choice(config.STRIDE)
            newNode.outputs = random.choice(config.NUMBER_OF_CONVOLUTION_FILTERS)
        else:
            newNode = self.addNode('linear')
            newNode.outputs = random.choice(config.NUMBER_OF_NODES)
        existingConnection = self.getRandomConnection()

        inputNode = getNode(existingConnection.inputNodeId)
        newNode.inputs = inputNode.outputs
        newNode.activation = random.choice(config.ACTIVATION)

        self.addConnection(existingConnection.inputNodeId, newNode.id)
        self.addConnection(newNode.id, existingConnection.outputNodeId)

        existingConnection.active = False

    def countExcessGenes(self, other):
        excessGeneCounter = 0

        innovationNumberMaximum = max(other.innovationNumbers)
        for connection in self.connections:
            if connection.innovationNumber > innovationNumberMaximum:
                excessGeneCounter += 1

        nodeIdMaximum = max([node.id for node in other.nodes])
        for node in self.nodes:
            if node.id > nodeIdMaximum:
                excessGeneCounter += 1

        return excessGeneCounter

    def countDisjointGenes(self, other):
        disjointgeneCounter = 0

        innovationNumberMaximum = max(other.innovationNumbers)
        for connection in self.connections:
            if connection.innovationNumber <= innovationNumberMaximum:
                if other.getConnection(connection.innovationNumber) is None:
                    disjointgeneCounter += 1

        nodeIdMaximum = max([node.id for node in other.nodes])
        for node in self.nodes:
            if node.id <= nodeIdMaximum:
                if other.getNode(node.id) is None:
                    disjointgeneCounter += 1

        return disjointgeneCounter

    def getConnection(self, innovationNumber):
        for connection in self.connections:
            if connection.innovationNumber == innovationNumber:
                return connection
        return None

    def getNode(self, nodeId):
        for node in self.nodes:
            if node.id == nodeId:
                return node
        return None

    def getAverageWeightDifference(self, other):
        weightDifference = 0.0
        weightNumber = 0.0

        for connection in self.connections:
            matchingConnection = other.getConnection(connection.innovationNumber)
            if matchingConnection is not None:
                weightDifference += float(connection.weight) - float(matchingConnection.weight)
                weightNumber += 1

        if weightNumber == 0.0:
            weightNumber = 1.0
        return weightDifference / weightNumber

    def getNodeInputIds(self, nodeId):
        nodeInputIds = []
        for connection in self.connections:
            if (connection.outputNodeId == nodeId) and connection.active:
                nodeInputIds.append(connection.inputNodeId)
        return nodeInputIds

    def addConnection(self, inputNodeId, outputNodeId, active=True, weight=None):
        newConnection = Connection(inputNodeId, outputNodeId, active)

        if weight is not None:
            newConnection.setWeight(float(weight))

        self.connections.append(newConnection)

        self.nodeIds.add(inputNodeId)
        self.nodeIds.add(outputNodeId)
        self.innovationNumbers.add(newConnection.innovationNumber)

    def addNode(self, nodeType):
        nodeId = len(self.nodes)
        node = Node(nodeId, nodeType)
        self.nodes.append(node)
        return node

    def addConnectionCopy(self, copy):
        newConnection = Connection(copy.inputNodeId, copy.outputNodeId, copy.active)
        newConnection.setWeight(float(copy.weight))
        newConnection.setInnovationNumber(copy.innovationNumber)

        self.connections.append(newConnection)

        self.nodeIds.add(copy.inputNodeId)
        self.nodeIds.add(copy.outputNodeId)
        self.innovationNumbers.add(newConnection.innovationNumber)

    def addNodeCopy(self, copy):
        self.nodes.append(Node(copy.id, copy.type))

    def getInputConnections(self, nodeId):
        connections = []
        for connection in self.connections:
            if (connection.outputNodeId == nodeId) and connection.active:
                connections.append(connection)
        return connections

    def getRandomNodeId(self):
        return random.choice(list(self.nodeIds))

    def getRandomConnection(self):
        return random.choice(self.connections)

    def getOutputConnections(self, nodeId):
        connections = []
        for connection in self.connections:
            if (connection.inputNodeId == nodeId) and connection.active:
                connections.append(connection)
        return connections

    def checkCyclicity(self, inputNodeId, outputNodeId):
        if inputNodeId == outputNodeId:
            return True

        visited = {outputNodeId}
        while True:
            counter = 0

            for connection in self.connections:
                if connection.inputNodeId in visited and connection.outputNodeId not in visited:

                    if connection.outputNodeId == inputNodeId:
                        return True
                    else:
                        visited.add(connection.outputNodeId)
                        counter += 1

            if counter == 0:
                return False

    def checkValidConnection(self, inputNodeId, outputNodeId):
        isThereCycle = self.checkCyclicity(inputNodeId, outputNodeId)
        exists = self.checkExistence(inputNodeId, outputNodeId)

        return (not isThereCycle) and (not exists)

    def checkExistence(self, inputNodeId, outputNodeId):
        for connection in self.connections:
            if (connection.inputNodeId == inputNodeId) and (connection.outputNodeId == outputNodeId):
                return True
            elif (connection.inputNodeId == outputNodeId) and (connection.outputNodeId == inputNodeId):
                return True
        return False

    def getOutputNodes(self, node, nodes):
        outputNodeIds = [connection.outputNodeId for connection in self.connections if (connection.inputNodeId == node.id) and connection.active]
        return [outputNode for outputNode in nodes if outputNode.id in outputNodeIds]

    def orderNodesByValue(self, nodeValues):

        nodes = [nodeValue.referenceNode for nodeValue in nodeValues]
        visited = set()
        ordered = []

        for node in nodes:
            if node not in visited:
                self.innerOrdering(node, nodes, ordered, visited)

        orderedNodeValues = []
        for node in ordered:
            for nodeValue in nodeValues:
                if nodeValue.referenceNode == node:
                    orderedNodeValues.append(nodeValue)
                    break

        return orderedNodeValues

    def innerOrdering(self, node, nodes, ordered, visited):
        visited.add(node)

        for outputNode in self.getOutputNodes(node, nodes):
            if outputNode not in visited:
                self.innerOrdering(outputNode, nodes, ordered, visited)

        ordered.append(node)