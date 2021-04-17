import gym
import torch
import DeepNEAT.NEAT_implementation.Population.population as population
import DeepNEAT.configurations.poleBalancing.poleBalancing as config
from DeepNEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
from DeepNEAT.visualization.Visualization import draw_net
import torch.nn as nn
import torch.nn.functional as F

configuration = config.PoleBalanceConfig()
if configuration.BUILD_TEST_DATA == True:
    configuration.build_test_data()
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

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(4, 4, False)
#         self.fc2 = nn.Linear(4, 2, False)
#
#     # x represents our data
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         # Apply softmax to x
#         output = F.log_softmax(x, dim=1)
#         return output
#
# random_data = torch.rand(100, 4)
#
# print(random_data)
# my_nn = Net()
# print(my_nn)
# result = my_nn(random_data)
# print (result)