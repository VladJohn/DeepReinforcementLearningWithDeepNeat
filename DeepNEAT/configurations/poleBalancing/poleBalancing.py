import torch
import gym
from DeepNEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time


class PoleBalanceConfig:
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    VERBOSE = True
    GAME = 'PoleBalancing'

    NUM_INPUTS = 1
    NUM_OUTPUTS = 1
    USE_BIAS = False
    INPUT_SIZE = 128
    OUTPUT_SIZE = 2

    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 10000.0

    POPULATION_SIZE = 15
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    # Node = layer hyper-parameters
    NUMBER_OF_NODES = [512, 1024, 2048]
    NUMBER_OF_NODES_MUTATION_RATE = 0.1
    NUMBER_OF_CONVOLUTION_FILTERS = [32, 64, 128]
    NUMBER_OF_CONVOLUTION_FILTERS_MUTATION_RATE = 0.3
    KERNEL_SIZE = [3, 4, 5, 6]
    KERNEL_SIZE_MUTATION_RATE = 0.3
    STRIDE = [1, 2]
    STRIDE_MUTATION_RATE = 0.3
    STRIDE_POOL = [1, 2]
    STRIDE_POOL_MUTATION_RATE = 0.3
    POOLSIZE = [2, 3]
    POOLSIZE_MUTATION_RATE = 0.3
    LAYER_TYPE = ['linear', 'conv1d']
    LAYER_TYPE_MUTATION_RATE = 0.2
    ACTIVATION = ['tanh', 'sigmoid', 'relu']
    ACTIVATION_MUTATION_RATE = 0.3
    MAXPOOL = [True, False]
    MAXPOOL_MUTATION_RATE = 0.2


    # Allow episode lengths of > than 200
    gym.envs.register(
        id='LongCartPole-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=10000
    )

    def fitness(self, genome):
        print ("Running genome...")
        # OpenAI Gym
        env = gym.make('LongCartPole-v0')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNetwork(genome, self)
        print('------------------------------------------------------------------')
        print(phenotype)
        print('------------------------------------------------------------------')

        while not done:
            #env.render()
            input = torch.Tensor([observation]).to(self.DEVICE)

            pred = round(float(phenotype(input)))
            observation, reward, done, info = env.step(pred)
            fitness += reward
        env.close()

        return fitness