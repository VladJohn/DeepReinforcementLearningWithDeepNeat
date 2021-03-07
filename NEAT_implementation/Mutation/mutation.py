import random
import NEAT_implementation.Utils.utils as utils

def mutate(genome, config):
    if utils.randomUniformValue() < config.CONNECTION_MUTATION_RATE:
        for connection in genome.connections:
            if utils.randomUniformValue() < config.CONNECTION_PERTURBATION_RATE:
                perturb = utils.randomUniformValue() * random.choice([1, -1])
                connection.weight += perturb
            else:
                connection.setWeightRandomly()

    if utils.randomUniformValue() < config.ADD_NODE_MUTATION_RATE:
        genome.addNodeMutation()

    if utils.randomUniformValue() < config.ADD_CONNECTION_MUTATION_RATE:
        genome.addConnectionMutation()
