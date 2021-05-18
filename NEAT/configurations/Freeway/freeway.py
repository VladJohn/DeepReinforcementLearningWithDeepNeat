import torch
import gym
from NEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork


class FreewayConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True
    GAME = 'Freeway'

    NUM_INPUTS = 128
    NUM_OUTPUTS = 3
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 20.0

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
        print ("Preparing genome...")
        # OpenAI Gym
        env = gym.make('Freeway-ram-v0')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNetwork(genome, self)

        print("Running genome...")
        while not done:
            # env.render()
            input = torch.Tensor([observation]).to(self.DEVICE)

            pred = torch.argmax(phenotype(input))
            observation, reward, done, info = env.step(pred)

            fitness += reward
        env.close()

        print("Done running genome... Returning to main...")
        return fitness