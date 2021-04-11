import gym
import torch
import NEAT.NEAT_implementation.Population.population as population
import NEAT.configurations.poleBalancing.poleBalancing as config
from NEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
from NEAT.visualization.Visualization import draw_net

configuration = config.PoleBalanceConfig()
neat = population.Population(configuration)
solution, generation = neat.run()

if solution is not None:
    # print('Found a Solution')
    draw_net(solution, view=True, filename='images/pole-balancing-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('LongCartPole-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNetwork(solution, config.PoleBalanceConfig)

    torch.save(solution, "./Results/" + neat.configuration.GAME + '/' +  neat.configuration.GAME +  '_final')

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(config.PoleBalanceConfig.DEVICE)

        pred = round(float(phenotype(input)))
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()