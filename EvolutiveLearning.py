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

    def __str__(self):
        return str(len(self.individuals)) + ": " + self.individuals[0].__str__()

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
            ind.fitness = np.mean(fitness) * min((1 - NEAT.norm_variance(np.var(fitness), np.min(fitness), np.max(fitness)) * 0.5), 1)
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
            ind.adj_fitness = ind.fitness / (sum(sh))
            self.current_adj_fitness.append(ind.adj_fitness)
            self.current_fitness.append(ind.fitness)

        self.top_fit = max(self.current_fitness)
        self.top_adj_fit = max(self.current_adj_fitness)
        self.top_fit_historic.append(self.top_fit)
        self.mean_fit = sum(self.current_adj_fitness) / len(self.current_adj_fitness)

        self.individuals, self.current_adj_fitness = zip(*sorted(zip(self.individuals, self.current_adj_fitness), key=lambda i: i[0].adj_fitness * -1))
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
        for i, ind in enumerate(self.individuals):
            if rand() < max(probs[i], 0.1):
                parents.append(ind)
        return parents
        pass

    def purge(self):
        survivors = []
        for i in self.individuals:
            if i.adj_fitness > self.top_adj_fit * (0.2 if self.top_adj_fit > 0 else 1.2):
                survivors.append(i)

        self.individuals = survivors


class Population:
    def __init__(self, n_individuals, inp, out):
        self.current_bests_inds = []
        self.n = n_individuals
        self.population = []
        self.species = []

        for ind in [Individual(inp, out) for i in range(n_individuals)]:
            ind.force_mutate()
            self.speciate(ind)

        # metrics
        self.means = []
        self.n_gens = 1
        self.best = self.species[0].individuals[0]
        self.bests = []
        self.current_gen_fitness = []
        self.mean_fitness = -1
        self.gen_fitness = []

    def mutate_individuals(self):
        for s in self.species:
            for ind in s.individuals:
                ind.mutate()
        # NEAT.innovations = {}

    def score(self):
        for s in self.species:
            s.score_individuals()
        self.mean_fitness = np.array([s.mean_fit for s in self.species]).mean()
        self.species = sorted(self.species, key=lambda s: s.individuals[0].fitness * -1)

    def nextGen(self):
        self.mutate_individuals()
        self.score()

        self.bests.append(self.species[0].individuals[0].fitness)
        self.means.append(self.mean_fitness)
        self.current_bests_inds = [s.individuals[0] for s in self.species]

        self.natural_selection()
        self.n_gens += 1
        pass

    def natural_selection(self):

        """
        25% primeres especies --> +10% penalització en nombre d'individus
        25% segons especies --> 0% penalització en nombre d'individus
        25% tercers especies --> 0% penalització
        25% ultims especies --> -0% penalització

        per fer el offspring agafarem els millors de cada especie i els reproduirem entre si
        --> Generara molta varietat genètica.

        """

        # purgar especies
        survivors = []
        for s in self.species:
            if s.mean_fit > max(self.means) * (0.2 if self.mean_fitness > 0 else 1.3):
                if len(s.individuals) > 10:
                    pass# s.purge()
                survivors.append(s)
                s.uselessnes = 0
            else:
                s.purge()
                s.uselessnes += 1
                if len(s.individuals) > 0 and s.uselessnes < 3:
                    survivors.append(s)

        self.species = survivors

        parents = []
        for s in self.species:
            parents += s.selectParents()
        """
        for s in self.species:
            border = int(np.ceil(len(s.individuals) / 4))
            for i in s.individuals[:border]:
                parents.append(i)
        """

        for i in parents:
            self.speciate(NEAT.crossover(i, choice(parents)))

        """
        for i in parents:
            i2 = choice(parents)
            c = NEAT.crossover(i, choice(parents))
            c.score()
            if c.fitness > i.fitness and c.fitness > i2.fitness:
                self.speciate(c)
        """

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
