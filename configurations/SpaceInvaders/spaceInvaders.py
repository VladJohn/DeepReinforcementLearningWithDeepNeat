import torch
import gym
from NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork


class SpaceInvadersConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True
    GAME = 'SpaceInvaders'

    NUM_INPUTS = 128
    NUM_OUTPUTS = 6
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 5000.0

    POPULATION_SIZE = 10
    NUMBER_OF_GENERATIONS = 15
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    def fitness(self, genome):
        print ("Running genome...")
        # OpenAI Gym
        env = gym.make('SpaceInvaders-ram-v0')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNetwork(genome, self)

        while not done:
            #env.render()
            input = torch.Tensor([observation]).to(self.DEVICE)

            pred = torch.argmax(phenotype(input))
            observation, reward, done, info = env.step(pred)

            fitness += reward
        env.close()

        return fitness