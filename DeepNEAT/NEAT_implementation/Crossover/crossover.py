from copy import deepcopy
import DeepNEAT.NEAT_implementation.Utils.utils as utils
from DeepNEAT.NEAT_implementation.Genotype.genome import Genome


def crossover(first, second, config):

    child = Genome()
    best, worse = orderParentsByFitness(first, second)

    for connection in best.connections:
        matchingConnection = worse.getConnection(connection.innovationNumber)

        if matchingConnection is not None:
            if utils.randomBoolean():
                childConnection = deepcopy(connection)
            else:
                childConnection = deepcopy(matchingConnection)
        else:
            childConnection = deepcopy(connection)

        if not childConnection.active:
            reactivate = utils.randomUniformValue() <= config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE
            reactivateInParent = best.getConnection(childConnection.innovationNumber).active

            if reactivate or reactivateInParent:
                childConnection.active = True

        child.addConnectionCopy(childConnection)

    for node in best.nodes:
        matchingNode = worse.getNode(node.id)

        if matchingNode is not None:
            if utils.randomBoolean():
                childNode = deepcopy(node)
            else:
                childNode = deepcopy(matchingNode)
        else:
            childNode = deepcopy(node)

        child.addNodeCopy(childNode)

    return child


def orderParentsByFitness(first, second):

    best = first
    worst = second

    firstLength = len(first.connections)
    secondLength = len(second.connections)

    if first.fitness == second.fitness:
        if firstLength == secondLength:
            if utils.randomBoolean():
                best = second
                worst = first

        elif secondLength < firstLength:
            best = second
            worst = first

    elif second.fitness > first.fitness:
        best = second
        worst = first

    return best, worst