import sys

class Species:

    def __init__(self, id, ancestor, generation):
        self.id = id
        self.ancestor = ancestor
        self.members = []
        self.fitnessHistory = []
        self.fitness = None
        self.adjustedFitness = None
        self.lastImproved = generation

    @staticmethod
    def speciesDistance(first, second):
        C1 = 1.0
        C2 = 1.0
        C3 = 0.5
        N = 1

        counterExcessGenes = first.countExcessGenes(second)
        counterDisjointGenes = first.countDisjointGenes(second)
        averageWeightDifference = first.getAverageWeightDifference(second)

        distance = (C1 * counterExcessGenes) / N
        distance += (C2 * counterDisjointGenes) / N
        distance += C3 * averageWeightDifference

        return distance

    @staticmethod
    def stagnation(species, generation):

        speciesData = []
        for specie in species:
            if len(specie.fitnessHistory) > 0:
                previousFitness = max(specie.fitnessHistory)
            else:
                previousFitness = -sys.float_info.max

            specie.fitness = max([network.fitness for network in specie.members])
            specie.fitnessHistory.append(specie.fitness)
            specie.adjustedFitness = None

            if previousFitness is None or specie.fitness > previousFitness:
                specie.lastImproved = generation

            speciesData.append(specie)

        speciesData.sort(key=lambda specie: specie.fitness)

        result = []
        speciesFitnesses = []
        countNotStagnantSpecies = len(speciesData)
        for i, specie in enumerate(speciesData):
            stagnantTime = generation - specie.lastImproved
            stagnant = False
            if countNotStagnantSpecies > 1:
                stagnant = stagnantTime >= 10

            if (len(speciesData) - i) <= 1:
                stagnant = False

            if stagnant:
                countNotStagnantSpecies -= 1

            result.append((specie, stagnant))
            speciesFitnesses.append(specie.fitness)

        return result