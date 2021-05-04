import gym
import torch
import DeepNEAT.NEAT_implementation.Population.population as population
import DeepNEAT.configurations.TimePilot.timePilot as config
from DeepNEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
from DeepNEAT.visualization.Visualization import draw_net
import torch.nn as nn
import torch.nn.functional as F

configuration = config.TimePilotConfig()
if configuration.BUILD_TEST_DATA == True:
    configuration.build_test_data()
neat = population.Population(configuration)
solution, generation = neat.run()

if solution is not None:
    # print('Found a Solution')
    draw_net(solution, view=True, filename='images/time-pilot-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('TimePilot-ram-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNetwork(solution, config.TimePilotConfig)

    torch.save(solution, "./Results/" + neat.configuration.GAME + '/' +  neat.configuration.GAME +  '_final')

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(config.TimePilotConfig.DEVICE)

        pred = torch.argmax(phenotype(input)[0]).numpy()
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()