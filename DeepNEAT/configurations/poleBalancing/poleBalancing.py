import torch
import statistics
import gym
from DeepNEAT.NEAT_implementation.Phenotype.feedForwardNetwork import FeedForwardNetwork
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm


class PoleBalanceConfig:
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    VERBOSE = True
    GAME = 'PoleBalancing'

    NUM_INPUTS = 1
    NUM_OUTPUTS = 1
    USE_BIAS = False
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2

    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 1000.0

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

    EPOCHS = 10
    BATCH_SIZE = 100
    BUILD_TEST_DATA = False
    LEARNING_RATE = 0.001
    VALIDATION_POINT = 0.1


    # Allow episode lengths of > than 200
    gym.envs.register(
        id='LongCartPole-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=10000
    )

    def fitness(self, genome):
        print("--------------------------------------------------")
        print ("Preparing genome...")
        # OpenAI Gym
        env = gym.make('LongCartPole-v0')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNetwork(genome, self)

        optimizer = optim.Adam(phenotype.parameters(), self.LEARNING_RATE)
        lossFunction = nn.MSELoss()

        trainingData = np.load('poleBalancing.npy', allow_pickle=True)
        X = torch.Tensor([i[0] for i in trainingData]).view(-1, self.INPUT_SIZE)
        y = torch.Tensor([i[1] for i in trainingData])

        valSize = int(len(X)*self.VALIDATION_POINT)

        trainX = X[:-valSize]
        trainy = y[:-valSize]

        testX = X[-valSize:]
        testy = y[-valSize:]

        print ("Training genome...")

        for epoch in range(self.EPOCHS):
            for i in range(0, len(trainX), self.BATCH_SIZE):
                batchX = trainX[i:i+self.BATCH_SIZE].view(-1, 4)
                batchy = trainy[i:i+self.BATCH_SIZE]

                phenotype.zero_grad()

                outputs = phenotype(batchX)
                loss = lossFunction(outputs, batchy)
                loss.backward()
                optimizer.step()

            #print(f"Epoch: {epoch}. Loss: {loss}")

        print ("Testing genome...")
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(testX)):
                real_class = torch.argmax(testy[i])
                net_out = phenotype(testX[i].view(-1, 4))[0]
                predicted_class = torch.argmax(net_out)
                if predicted_class == real_class:
                    correct += 1
                total += 1
        print("Accuracy: ", round(correct/total, 3))

        print ("Running genome...")
        while not done:
            #env.render()
            input = torch.Tensor([observation]).to(self.DEVICE)

            pred = torch.argmax(phenotype(input)[0]).numpy()
            observation, reward, done, info = env.step(pred)
            fitness += reward
        env.close()

        return fitness

    def build_test_data(self):
        print ('Bulding test data...')
        env = gym.make("LongCartPole-v0")
        env.reset()
        goalSteps = 500
        scoreRequirement = 50
        initialGames = 10000
        trainingData = []
        scores = []
        acceptedScores = []
        for _ in range(initialGames):
            score = 0
            gameMemory = []
            prevObservation = []
            for _ in range(goalSteps):
                action = random.randrange(0,2)
                observation, reward, done, info = env.step(action)
                if len(prevObservation) > 0 :
                    gameMemory.append([prevObservation, action])
                prevObservation = observation
                score+=reward
                if done: break

            if score >= scoreRequirement:
                acceptedScores.append(score)
                for data in gameMemory:
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]
                    trainingData.append([data[0], output])
            env.reset()
            scores.append(score)
        trainingDataSave = np.array(trainingData)
        np.save('poleBalancing.npy', trainingDataSave)
        print('Average accepted score:', statistics.mean(acceptedScores))
        print('Median score for accepted scores:', statistics.median(acceptedScores))
        print(len(acceptedScores))
        return trainingData