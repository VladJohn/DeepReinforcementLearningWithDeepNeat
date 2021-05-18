import gym
import torch
import DeepNEAT.NEAT_implementation.Population.population as population
import DeepNEAT.configurations.SpaceInvaders.spaceInvaders as config
from DeepNEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
from DeepNEAT.visualization.Visualization import draw_net
import torch.nn as nn
import torch.nn.functional as F

configuration = config.SpaceInvadersConfig()
neat = population.Population(configuration)
solution, generation = neat.run()

if solution is not None:
    # print('Found a Solution')
    draw_net(solution, view=True, filename='./images/space-invaders-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('SpaceInvaders-ram-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNetwork(solution, config.SpaceInvadersConfig)

    torch.save(solution, "./Results/" + neat.configuration.GAME + '/' +  neat.configuration.GAME +  '_final')

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(config.SpaceInvadersConfig.DEVICE)

        pred = torch.argmax(phenotype(input)[0]).numpy()
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()