import gym
import torch
import NEAT_implementation.Population.population as population
import configurations.TimePilot.timePilot as config
#import configurations.Freeway.freeway as config
#import configurations.SpaceInvaders.spaceInvaders as config
from NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
from visualization.Visualization import draw_net
from NEAT_implementation.Genotype.genome import Genome

configuration = config.TimePilotConfig()
#configuration = config.FreewayConfig()
#configuration = config.SpaceInvadersConfig()
# OpenAI Gym
env = gym.make('TimePilot-ram-v0')
#env = gym.make('Freeway-ram-v0')
#env = gym.make('SpaceInvaders-ram-v0')
done = False
observation = env.reset()

fitness = 0
solution = torch.load("./Results/TimePilot/TimePilot_11")
#solution = torch.load("./Results/Freeway/Freeway_14")
#solution = torch.load("./Results/SpaceInvaders/SpaceInvaders_11")
phenotype = FeedForwardNetwork(solution, config.TimePilotConfig)
#phenotype = FeedForwardNetwork(solution, config.FreewayConfig)
#phenotype = FeedForwardNetwork(solution, config.SpaceInvadersConfig)
while not done:
    env.render()
    input = torch.Tensor([observation]).to(config.TimePilotConfig.DEVICE)
    #input = torch.Tensor([observation]).to(config.FreewayConfig.DEVICE)
    #input = torch.Tensor([observation]).to(config.SpaceInvadersConfig.DEVICE)

    pred = torch.argmax(phenotype(input))
    observation, reward, done, info = env.step(pred)

    fitness += reward
env.close()