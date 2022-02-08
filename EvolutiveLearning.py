import names
from Individual import *


class Specie:

    def __init__(self, ind):
        self.top_adj_fit = 0
        self.individuals = [ind]
        self.adj_fitness = 0
        self.top_fit = -1
        self.current_adj_fitness = []
        self.current_fitness = []
        self.mean_fit = 0
        self.uselessnes = 0
        self.top_fit_historic = []

        self.name = names.get_last_name()

    def __str__(self):
        return "{0:<15}".format(self.name) + "\t\tMean: " + "{0:<4}".format(
            str(self.mean_fit.__round__(2))) + " \t\tBest: " + str(
            self.current_adj_fitness[0].__round__(2)) + " \t\tNº:" + str(
            len(self.individuals)) + " \t\tUselessness: " + str(self.uselessnes) + " \t\tTop FIT: " + str(
            max(self.top_fit_historic)) + "\n"

    def calcDistance(self, g2):
        g1 = choice(self.individuals)
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
        N = 1 if NEAT.max_len < 20 else NEAT.max_len
        distance = NEAT.c1 * (excess / N) + NEAT.c2 * (disjoint / N) + NEAT.c3 * (w_mean)
        return distance
        pass

    def score_individuals(self):
        self.current_fitness = []
        self.current_adj_fitness = []
        # Calc fitness
        for ind in self.individuals:
            fitness = []
            trys = 1
            for i in range(trys):
                fitness.append(NEAT.fitness(ind))
            ind.fitness = np.mean(fitness) * min(
                (1 - NEAT.norm_variance(np.var(fitness), np.min(fitness), np.max(fitness)) * 0.5), 1)
            continue

        # Adjust fitness (explicit fitness sharing)
        for ind in self.individuals:
            sh = []
            for bro in self.individuals:
                dist = ind.calcDistance(bro)
                thld = NEAT.distance_thld - len(self.individuals) / 15
                if dist > thld:
                    sh.append(0)
                else:
                    if dist == 0:
                        sh.append(1)
                    else:
                        sh.append(1 - (ind.calcDistance(bro) / thld))

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
        self.top_fit_historic.append(self.top_fit)
        self.mean_fit = sum(self.current_adj_fitness) / len(self.current_adj_fitness)

        self.individuals, self.current_adj_fitness = zip(
            *sorted(zip(self.individuals, self.current_adj_fitness), key=lambda i: i[0].adj_fitness * -1))
        self.individuals = list(self.individuals)
        self.current_adj_fitness = list(self.current_adj_fitness)
        pass

    def get_offspring(self, n):
        children = []
        for i in range(int(n)):
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


class Population:
    def __init__(self, n_individuals, inp, out):
        self.gen_bests = []
        self.n = n_individuals
        self.species = []

        for ind in [Individual(inp, out) for i in range(n_individuals)]:
            ind.force_mutate()
            self.speciate(ind)
        # metrics
        self.means = []
        self.n_gens = 1
        self.best = self.species[0].individuals[0]
        self.bests = [self.best.fitness]
        self.current_gen_fitness = []
        self.mean_fitness = -1
        self.gen_fitness = []

        self.file = open("report_" + str(int(np.ceil(rand() * 100))) + ".txt", "w")

    def __str__(self):
        out = ""
        out += "########## REPORT GEN: " + str(self.n_gens) + " ##########\n"
        out += "SPECIES: \n"
        for i in self.species:
            if len(i.current_adj_fitness) > 1:
                out += i.__str__()
        out += "\n"
        return out

    def mutate_individuals(self):
        for s in self.species:
            for ind in s.individuals:
                ind.mutate()
        NEAT.innovations = {}

    def score(self):
        for s in self.species:
            s.score_individuals()
        self.mean_fitness = np.array([s.mean_fit for s in self.species]).mean()
        self.species = sorted(self.species, key=lambda s: s.individuals[0].fitness * -1)

    def metrics(self):
        if self.n_gens > 1:
            self.file.write(self.__str__())
        self.n_gens += 1

        top = self.species[0].individuals[0]
        if top.fitness > self.best.fitness:
            self.best = IndividualFactory.buildIndividual(len(top.inp), len(top.out), top.genome.values())
            self.best.fitness = top.fitness
            self.best.adj_fitness = top.adj_fitness
        if self.bests[-1] < self.best.fitness and rand() < 0.5:
            self.speciate(self.best)

    def nextGen(self):
        if sum(self.bests[-4:]) >= NEAT.max_rwd * 4:
            return

        if len(self.best.hidden) > 10:
            NEAT.new_node_mutation_rate = 0.3
            NEAT.new_link_mutation_rate = 0.5
            NEAT.step = 0.7

        if len(self.best.hidden) > 15:
            NEAT.new_node_mutation_rate = 0.1
            NEAT.new_link_mutation_rate = 0.4
            NEAT.c1 = 0.35
            NEAT.c2 = 0.35
            NEAT.c3 = 0.3

        self.mutate_individuals()
        self.score()

        self.gen_bests.append(self.species[0].individuals[0].fitness)
        self.means.append(self.mean_fitness)

        self.natural_selection()

        self.metrics()
        pass

    def natural_selection(self):

        # purgar especies
        survivors = []
        fallen = []
        for s in self.species:
            if s.mean_fit > max(self.means) * (0.7 if self.mean_fitness > 0 else 1.3):
                if len(s.individuals) > 5:
                    s.purge()
                survivors.append(s)
                s.uselessnes = 0
            else:
                s.purge()
                s.uselessnes += 1
                if len(s.individuals) > 0 and s.uselessnes < 5:
                    survivors.append(s)
                else:
                    fallen.append(s)

        self.species = survivors

        parents = []
        for s in self.species:
            parents += s.selectParents()

        for s in fallen:
            if len(s.individuals) > 0:
                self.migrate_individual(s.individuals[0])

        for i in parents:
            self.speciate(NEAT.crossover(i, choice(parents)))

    def speciate(self, ind):
        l = copy.copy(self.species)
        while len(l) > 0:
            c = choice(l)
            if c.calcDistance(ind) < NEAT.distance_thld:
                c.individuals.append(ind)
                break
            else:
                l.remove(c)
        if len(l) == 0:
            self.species.append(Specie(ind))

    def migrate(self):
        if len(self.species) < 2:
            return

        for s in self.species:
            s2 = choice(self.species)
            if s2 is not s:
                s2.individuals.append(copy.deepcopy(choice(s.individuals)))

    def migrate_individual(self, ind):

        d = np.zeros(len(self.species))
        for i, s in enumerate(self.species):
            d[i] = s.calcDistance(ind)

        m = np.argmin(d)
        self.species[m].individuals.append(ind)
