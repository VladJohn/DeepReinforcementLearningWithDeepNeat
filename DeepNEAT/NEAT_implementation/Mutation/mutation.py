import random
import DeepNEAT.NEAT_implementation.Utils.utils as utils

def mutate(genome, config):
    if utils.randomUniformValue() < config.CONNECTION_MUTATION_RATE:
        for connection in genome.connections:
            if utils.randomUniformValue() < config.CONNECTION_PERTURBATION_RATE:
                perturb = utils.randomUniformValue() * random.choice([1, -1])
                connection.weight += perturb
            else:
                connection.setWeightRandomly()

    if utils.randomUniformValue() < config.ADD_NODE_MUTATION_RATE:
        genome.addNodeMutation(config)

    if utils.randomUniformValue() < config.NUMBER_OF_NODES_MUTATION_RATE:
        for node in genome.nodes:
            if (node.type == 'linear' and utils.randomUniformValue() < config.NUMBER_OF_NODES_MUTATION_RATE):
                newNode.outputs = random.choice(config.NUMBER_OF_NODES)

    if utils.randomUniformValue() < config.NUMBER_OF_CONVOLUTION_FILTERS_MUTATION_RATE:
        for node in genome.nodes:
            if (node.type == 'conv1d' and utils.randomUniformValue() < config.NUMBER_OF_CONVOLUTION_FILTERS_MUTATION_RAT):
                newNode.outputs = random.choice(config.NUMBER_OF_CONVOLUTION_FILTERS)

    if utils.randomUniformValue() < config.KERNEL_SIZE_MUTATION_RATE:
        for node in genome.nodes:
            if (node.type == 'conv1d' and utils.randomUniformValue() < config.KERNEL_SIZE_MUTATION_RATE):
                node.kernelSize = random.choice(config.KERNEL_SIZE)

    if utils.randomUniformValue() < config.STRIDE_MUTATION_RATE:
        for node in genome.nodes:
            if (node.type == 'conv1d' and utils.randomUniformValue() < config.STRIDE_MUTATION_RATE):
                node.stride = random.choice(config.STRIDE)

    if utils.randomUniformValue() < config.LAYER_TYPE_MUTATION_RATE:
        for node in genome.nodes:
            if (node.type == 'linear' and utils.randomUniformValue() < config.LAYER_TYPE_MUTATION_RATE):
                node.type = 'conv1d'
                node.kernelSize = random.choice(config.KERNEL_SIZE)
                node.stride = random.choice(config.STRIDE)
                node.outputs = random.choice(config.NUMBER_OF_CONVOLUTION_FILTERS)
            elif (node.type == 'conv1d' and utils.randomUniformValue() < config.LAYER_TYPE_MUTATION_RATE):
                node.type = 'lnear'
                node.outputs = random.choice(config.NUMBER_OF_NODES)

    if utils.randomUniformValue() < config.ACTIVATION_MUTATION_RATE:
        for node in genome.nodes:
            node.activation = random.choice(config.ACTIVATION)

    # if utils.randomUniformValue() < config.ADD_CONNECTION_MUTATION_RATE:
    #     genome.addConnectionMutation()
