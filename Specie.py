import copy
import names
import numpy as np
from NEAT import NEAT
from numpy.random import choice, rand


class Specie:

    def __init__(self, ind):
        self.uselessness = 0
        self.top_adj_fit = 0
        self.individuals = [ind]
        self.adj_fitness = 0
        self.top_fit = -1
        self.current_adj_fitness = []
        self.current_fitness = []
        self.mean_fit = 0
        self.top_fit_historic = []

        self.leader = copy.deepcopy(ind)
        self.name = names.get_last_name()

    def __str__(self):
        return "{0:<15}".format(self.name) + "\t\tMean: " + "{0:<4}".format(
            str(self.mean_fit.__round__(2))) + " \t\tBest: " + str(
            self.top_fit.__round__(2)) + " \t\tN:" + str(
            len(self.individuals)) + "\n"

    def calcDistance(self, g2):
        g1 = self.leader
        k1 = sorted(g1.genome.keys())
        k2 = sorted(g2.genome.keys())
        border = min(max(k1), max(k2))
        set1 = set(k1)
        set2 = set(k2)

        disjoint1 = set1.difference(set2)
        disjoint2 = set2.difference(set1)
        differ = np.array(list(disjoint1.union(disjoint2)))

        disjoint = len(differ[differ <= border])
        excess = len(differ) - disjoint

        w_mean = 0
        intersec = list(set1.intersection(set2))
        if len(intersec) == 0:
            return np.inf

        for g in intersec:
            w_mean += abs(g1.genome[g].w - g2.genome[g].w)

        w_mean /= len(intersec)
        N = 1 if NEAT.max_len < 20 else 1 - NEAT.max_len * 0.005
        try:
            distance = NEAT.c1 * (excess / N) + NEAT.c2 * (disjoint / N) + NEAT.c3 * w_mean
        except:
            distance = np.inf

        return distance
        pass

    def score_individuals(self):
        if len(self.individuals) == 0:
            return
        self.current_fitness = []
        self.current_adj_fitness = []
        # Calc fitness
        for ind in self.individuals:
            try:
                fitness = []
                trys = NEAT.reps
                for i in range(trys):
                    fitness.append(NEAT.fitness(ind))
                ind.fitness = np.mean(fitness) * min(
                    (1 - NEAT.norm_variance(np.var(fitness), np.min(fitness), np.max(fitness)) * 0.5), 1)
            except:
                continue

        # Adjust fitness (explicit fitness sharing)
        for ind in self.individuals:
            sh = []
            for bro in self.individuals:
                dist = ind.calcDistance(bro)
                if dist > NEAT.distance_thld:
                    sh.append(0)
                else:
                    if dist == 0:
                        sh.append(1)
                    else:
                        sh.append(1 - (ind.calcDistance(bro) / NEAT.distance_thld))

            if sum(sh) == 0:
                ind.adj_fitness = ind.fitness
            elif ind.fitness > 0:
                ind.adj_fitness = ind.fitness / (sum(sh))
            else:
                ind.adj_fitness = ind.fitness - ind.fitness * -1 / (sum(sh))

            self.current_adj_fitness.append(ind.adj_fitness)
            self.current_fitness.append(ind.fitness)

        self.top_fit = max(self.current_fitness)
        self.top_adj_fit = max(self.current_adj_fitness)
        if len(self.top_fit_historic) == 0:
            self.top_fit_historic.append(self.top_fit)
        if self.top_fit > self.top_fit_historic[-1]:
            self.uselessness = 0
            self.top_fit_historic.append(self.top_fit)
        else:
            self.uselessness += 1

        self.top_fit_historic.append(self.top_fit)
        self.mean_fit = sum(self.current_adj_fitness) / len(self.current_adj_fitness)

        self.individuals, self.current_adj_fitness = zip(
            *sorted(zip(self.individuals, self.current_adj_fitness),
                    key=lambda i: i[0].adj_fitness * -1 if NEAT.opt == "max" else 1))
        self.individuals = list(self.individuals)
        self.current_adj_fitness = np.array(list(self.current_adj_fitness))

        self.leader = choice(self.individuals)
        return self.current_adj_fitness
        pass

    def get_offspring(self, n):
        champion = copy.deepcopy(self.individuals[0])
        children = [champion]
        f = -1
        if len(self.individuals) > 15:
            f = int(len(self.individuals) / 5)
        for i in range(int(n - 1)):
            if f != -1:
                ind1 = choice(self.individuals[:f])
            else:
                ind1 = choice(self.individuals)
            ind2 = choice(self.individuals)
            child = NEAT.crossover(ind1, ind2)
            child.mutate()
            children.append(child)
        return children

    def selectParents(self):
        parents = []
        probs = NEAT.normalize(np.array(self.current_adj_fitness))
        probs[probs < 0.8] = 0.1
        for i, ind in enumerate(self.individuals):
            if rand() < max(probs[i], 0.1):
                parents.append(ind)
        return parents
        pass

    def purge(self):
        survivors = []
        for i in self.individuals:
            if i.adj_fitness > self.top_adj_fit * (0.5 if self.top_adj_fit > 0 else 1.5):
                survivors.append(i)

        self.individuals = survivors

        if len(self.individuals) > 10:
            self.individuals = self.individuals[:10]

    def reset(self):
        self.individuals = []

