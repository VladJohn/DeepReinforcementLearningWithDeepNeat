import gym
import torch
import DeepNEAT.NEAT_implementation.Population.population as population
import DeepNEAT.configurations.Freeway.freeway as config
from DeepNEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
from DeepNEAT.visualization.Visualization import draw_net

configuration = config.FreewayConfig()
if configuration.BUILD_TEST_DATA == True:
    configuration.build_test_data()
neat = population.Population(configuration)
solution, generation = neat.run()

if solution is not None:
    # print('Found a Solution')
    draw_net(solution, view=True, filename='images/freeway-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('Freeway-ram-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNetwork(solution, config.FreewayConfig)

    torch.save(solution, "./Results/" + neat.configuration.GAME + '/' +  neat.configuration.GAME +  '_final')

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(config.FreewayConfig.DEVICE)

        pred = torch.argmax(phenotype(input)[0]).numpy()
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()