import torch
import DeepNEAT.NEAT_implementation.Population.population as population

DEVICE = torch.device("cpu")

class Connection:
    def __init__(self, inputNodeId, outputNodeId, active):
        self.inputNodeId = inputNodeId
        self.outputNodeId = outputNodeId
        self.active = active
        self.innovationNumber = self.getInnovationNumber()
        self.weight = None
        self.setWeightRandomly()

    def setWeight(self, weight):
        self.weight = torch.Tensor([weight]).to(DEVICE)

    def setWeightRandomly(self):
        self.weight = torch.Tensor(torch.normal(torch.arange(0, 1).float())).to(DEVICE)

    def setInnovationNumber(self, number):
        self.innovationNumber = number

    def getInnovationNumber(self):
        for gene in population.Population.currentGenerationInnovation:
            if self == gene:
                return gene.innovationNumber

        population.Population.currentGenerationInnovation.append(self)
        return population.Population.getNewInnovationNumber()

    def __eq__(self, other):
        return (self.inputNodeId == other.inputNodeId) and (self.outputNodeId == other.outputNodeId)