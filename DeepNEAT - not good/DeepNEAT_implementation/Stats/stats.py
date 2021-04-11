import copy
import csv

from statistics import mean, stdev, median
from DeepNEAT.DeepNEAT_implementation.Reporting.reporting import BaseReporter
from six import iteritems

class StatisticsReporter(BaseReporter):
    def __init__(self):
        BaseReporter.__init__(self)
        self.most_fit_genomes = []
        self.generation_statistics = []

    def post_evaluate(self, config, population, species, best_genome):
        self.most_fit_genomes.append(copy.deepcopy(best_genome))

        species_stats = {}

        for sid, s in iteritems(species.species):
            species_stats[sid] = dict((k, v.fitness) for k, v in iteritems(s.members))
        self.generation_statistics.append(species_stats)

    def get_fitness_stat(self, f):
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(f(scores))

        return stat

    def get_fitness_mean(self):
        return self.get_fitness_stat(mean)

    def get_fitness_stdev(self):
        return self.get_fitness_stat(stdev)

    def get_fitness_median(self):
        return self.get_fitness_stat(median)

    def get_average_cross_validation_fitness(self):
        avg_cross_validation_fitness = []
        for stats in self.generation_cross_validation_statistics:
            scores = []
            for fitness in stats.values():
                scores.extend(fitness)
            avg_cross_validation_fitness.append(mean(scores))

        return avg_cross_validation_fitness

    def best_unique_genomes(self, n):
        best_unique = {}
        for g in self.most_fit_genomes:
            best_unique[g.key] = g
        best_unique_list = list(best_unique.values())

        def key(genome):
            return genome.fitness

        return sorted(best_unique_list, key=key, reverse=True)[:n]

    def best_genomes(self, n):
        def key(g):
            return g.fitness

        return sorted(self.most_fit_genomes, key=key, reverse=True)[:n]

    def best_genome(self):
        return self.best_genomes(1)[0]

    def save(self):
        self.save_genome_fitness()
        self.save_species_count()
        self.save_species_fitness()

    def save_genome_fitness(self,
                            delimiter=' ',
                            filename='fitness_history.csv',
                            with_cross_validation=False):
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)

            best_fitness = [c.fitness for c in self.most_fit_genomes]
            avg_fitness = self.get_fitness_mean()

            if with_cross_validation: # pragma: no cover
                cv_best_fitness = [c.cross_fitness for c in self.most_fit_genomes]
                cv_avg_fitness = self.get_average_cross_validation_fitness()
                for best, avg, cv_best, cv_avg in zip(best_fitness,
                                                      avg_fitness,
                                                      cv_best_fitness,
                                                      cv_avg_fitness):
                    w.writerow([best, avg, cv_best, cv_avg])
            else:
                for best, avg in zip(best_fitness, avg_fitness):
                    w.writerow([best, avg])

    def save_species_count(self, delimiter=' ', filename='speciation.csv'):
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)
            for s in self.get_species_sizes():
                w.writerow(s)

    def save_species_fitness(self, delimiter=' ', null_value='NA', filename='species_fitness.csv'):
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)
            for s in self.get_species_fitness(null_value):
                w.writerow(s)

    def get_species_sizes(self):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_counts = []
        for gen_data in self.generation_statistics:
            species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts

    def get_species_fitness(self, null_value=''):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_fitness = []
        for gen_data in self.generation_statistics:
            member_fitness = [gen_data.get(sid, []) for sid in range(1, max_species + 1)]
            fitness = []
            for mf in member_fitness:
                if mf:
                    fitness.append(mean(mf))
                else:
                    fitness.append(null_value)
            species_fitness.append(fitness)

        return species_fitness
