import gym
import torch
import NEAT_implementation.Population.population as population
import configurations.mountainClimbing.mountainClimbing as config
from NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
from visualization.Visualization import draw_net

configuration = config.MountainClimbConfig()
neat = population.Population(configuration)
solution, generation = neat.run()

if solution is not None:
    # print('Found a Solution')
    draw_net(solution, view=True, filename='./images/mountain-climb-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('MountainCarContinuous-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNetwork(solution, config.MountainClimbConfig)

    torch.save(solution, "./Results/" + neat.configuration.GAME + '/' +  neat.configuration.GAME +  '_final')

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(config.MountainClimbConfig.DEVICE)

        pred = [round(float(phenotype(input)))]
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()