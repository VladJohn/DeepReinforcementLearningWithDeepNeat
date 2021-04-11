import random
import numpy as np
import torch
import DeepNEAT.NEAT_implementation.Utils.utils as utils
from DeepNEAT.NEAT_implementation.Genotype.genome import Genome
from DeepNEAT.NEAT_implementation.Species.species import Species
from DeepNEAT.NEAT_implementation.Crossover.crossover import crossover
from DeepNEAT.NEAT_implementation.Mutation.mutation import mutate


class Population:
    globalInovationNumber = 0
    currentGenerationInnovation = []

    def __init__(self, configuration):
        self.configuration = configuration
        self.population = self.setInitialPopulation()
        self.species = []

        for genome in self.population:
            self.speciate(genome, 0)

    def run(self):
        for generation in range(1, self.configuration.NUMBER_OF_GENERATIONS):
            for genome in self.population:
                genome.fitness = max(0, self.configuration.fitness(genome))

            best = utils.getBestPerformingNetwork(self.population)

            allFitnesses = []
            remainingSpecies = []

            for species, stagnant in Species.stagnation(self.species, generation):
                if stagnant:
                    self.species.remove(species)
                else:
                    allFitnesses.extend(network.fitness for network in species.members)
                    remainingSpecies.append(species)

            fitnessMinimum = min(allFitnesses)
            fitnessMaximum = max(allFitnesses)

            fitnessRange = max(1.0, (fitnessMaximum - fitnessMinimum))
            for species in remainingSpecies:

                averageSpecieFitness = np.mean([network.fitness for network in species.members])
                species.adjustedFitness = (averageSpecieFitness - fitnessMinimum) / fitnessRange

            adjustedFitness = [specie.adjustedFitness for specie in remainingSpecies]
            sumOfAdjustedFitness = sum(adjustedFitness)

            freshPopulation = []
            for species in remainingSpecies:
                if species.adjustedFitness > 0:
                    size = max(2, int((species.adjustedFitness/sumOfAdjustedFitness) * self.configuration.POPULATION_SIZE))
                else:
                    size = 2

                specieMembers = species.members
                specieMembers.sort(key=lambda network: network.fitness, reverse=True)
                species.members = []  # reset

                freshPopulation.append(specieMembers[0])
                size -= 1

                pruneIndex = int(self.configuration.PERCENTAGE_TO_SAVE * len(specieMembers))
                pruneIndex = max(2, pruneIndex)
                specieMembers = specieMembers[:pruneIndex]

                for i in range(size):
                    first = random.choice(specieMembers)
                    second = random.choice(specieMembers)

                    child = crossover(first, second, self.configuration)
                    mutate(child, self.configuration)
                    freshPopulation.append(child)

            self.population = freshPopulation
            Population.currentGenerationInnovation = []

            # Speciate
            for genome in self.population:
                self.speciate(genome, generation)

            if best.fitness >= self.configuration.FITNESS_THRESHOLD:
                return best, generation

            # Generation Stats
            if self.configuration.VERBOSE:
                print('Finished Generation',  generation)
                print('Best Genome Fitness:', best.fitness)
                print('Best Genome Length',   len(best.connections))
                print()

            torch.save(best, "./Results/" + self.configuration.GAME + '/' +  self.configuration.GAME +  '_' + str(generation))

        return None, None

    def speciate(self, genome, generation):
        for species in self.species:
            if Species.speciesDistance(genome, species.ancestor) <= self.configuration.SPECIATION_THRESHOLD:
                genome.species = species.id
                species.members.append(genome)
                return

        freshSpecies = Species(len(self.species), genome, generation)
        genome.species = freshSpecies.id
        freshSpecies.members.append(genome)
        self.species.append(freshSpecies)

    def assignNewAncestors(self, species):
        speciesPopulation = self.getGenomesInSpecies(species.id)
        species.ancestor = random.choice(speciesPopulation)

    def getGenomesInSpecies(self, speciesId):
        return [specie for specie in self.population if specie.species == speciesId]

    def setInitialPopulation(self):
        population = []
        for i in range(self.configuration.POPULATION_SIZE):
            freshGenome = Genome()
            input = None
            output = None
            # bias = None

            input = freshGenome.addNode('input')
            input.outputs = self.configuration.INPUT_SIZE

            output = freshGenome.addNode('output')
            output.inputs = self.configuration.OUTPUT_SIZE

            # if self.configuration.USE_BIAS:
            #     bias = freshGenome.addNode('bias')

            freshGenome.addConnection(input.id, output.id)

            # if bias is not None:
            #     for output in outputs:
            #         freshGenome.addConnection(bias.id, output.id)

            population.append(freshGenome)

        return population

    @staticmethod
    def getNewInnovationNumber():
        result = Population.globalInovationNumber
        Population.globalInovationNumber += 1
        return result
