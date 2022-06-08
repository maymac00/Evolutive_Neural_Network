import copy

from Specie import Specie
from Individual import *
from IndividualFactory import *
import NEAT


class Population:
    def __init__(self, n_individuals, inp, out):
        self.total_fitness = []
        self.gen_bests = []
        self.n = n_individuals
        self.species = []

        for ind in [Individual(inp, out) for i in range(n_individuals)]:
            for i in ind.inp:
                for o in ind.out:
                    ind.add_connection(i, o, True)
            for i in range(6):
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
        NEAT.NEAT.innovations = {}

    def score(self):
        self.total_fitness = []
        for s in self.species:
            f = s.score_individuals()

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
        try:
            b = all([self.bests[:-(i + 1)] < self.best.fitness for i in range(3)])
            if len(self.bests) > 5 and not b:
                self.speciate(copy.deepcopy(self.best))
            self.bests.append(self.best.fitness)
        except:
            pass

    def nextGen(self):
        if sum(self.bests[-4:]) >= NEAT.NEAT.max_rwd * 4:
            return

        self.mutate_individuals()
        self.score()

        self.gen_bests.append(self.species[0].individuals[0].fitness)
        self.means.append(self.mean_fitness)

        print(self)
        self.metrics()

        self.natural_selection()

        # Step decay
        NEAT.NEAT.step -= 0.02
        NEAT.NEAT.step = max(NEAT.NEAT.step, 1)

        if len(self.species) < NEAT.NEAT.species_pool_size * 1.2:
            NEAT.NEAT.distance_thld -= NEAT.NEAT.adaptation
            print("Necessitem més especies", NEAT.NEAT.distance_thld)
        elif len(self.species) > NEAT.NEAT.species_pool_size * 0.8:
            NEAT.NEAT.distance_thld += NEAT.NEAT.adaptation * max((len(self.species) - NEAT.NEAT.species_pool_size) / 3, 1)
            print("Necessitem menys especies", NEAT.NEAT.distance_thld)
        if NEAT.NEAT.distance_thld < 0.3:
            NEAT.NEAT.distance_thld = 0.3

        self.speciate(self.best)
        pass

    def natural_selection(self):

        P = self.n  # self.calcSize()

        S = sum([s.current_adj_fitness.mean() for s in self.species])

        self.species = [s for s in self.species if len(s.individuals) > 0]
        spc = copy.copy(self.species)
        migrations = []
        to_speciate = []
        for s in spc:
            n = (np.array(s.current_adj_fitness).mean() / S) * P
            if n > len(s.individuals) * 2:
                n = len(s.individuals) * 2
            if n < len(s.individuals) * 0.5:
                n = len(s.individuals) * 0.5
            if n > 0 and s.uselessness < NEAT.NEAT.dropoff:
                c = s.get_offspring(n)
                s.individuals = []
                for ind in c:
                    to_speciate.append(ind)
            else:
                migrations.append(s.individuals[0])
                s.individuals = []

        cont = 0
        n_species = len([s for s in self.species if len(s.individuals) > 0])
        while (n_species < NEAT.NEAT.species_pool_size * 0.8 or n_species > NEAT.NEAT.species_pool_size * 1.2) and cont < 10:
            for s in self.species:
                s.reset()
            for ind in to_speciate + migrations:
                self.speciate(ind)
            if len(self.species) < NEAT.NEAT.species_pool_size * 1.2:
                NEAT.NEAT.distance_thld -= 0.3
            elif len(self.species) > NEAT.NEAT.species_pool_size * 0.8:
                NEAT.NEAT.distance_thld += 0.3
            if NEAT.NEAT.distance_thld < 0.3:
                NEAT.NEAT.distance_thld = 0.3
            n_species = len([s for s in self.species if len(s.individuals) > 0])
            cont += 1

        self.species = [s for s in self.species if len(s.individuals) > 0]
        for m in migrations:
            self.speciate(m)
        print(sum([len(s.individuals) for s in self.species]), len(self.species))
        pass

    def speciate(self, ind, prev=None):
        l = copy.copy(self.species)
        if prev is not None:
            d = prev.calcDistance(ind)

            if d < NEAT.NEAT.distance_thld * NEAT.NEAT.blood_rate:
                prev.individuals.append(ind)
                return
            else:
                l.remove(prev)
        while len(l) > 0:
            c = choice(l)
            d = c.calcDistance(ind)

            if d < NEAT.NEAT.distance_thld:
                c.individuals.append(ind)
                break
            else:
                l.remove(c)
        if len(l) == 0:
            self.species.append(Specie(ind))
            print("New specie")

    def calcSize(self):
        a = self.n / 3
        G = 10
        g = self.n_gens
        return int(self.n + a - (2 * a / (G - 1)) * (g - G * int((g - 1) / G) - 1))
        pass

    def purge_species(self):
        l = []
        misc = []
        for s in copy.copy(self.species):
            if len(s.individuals) == 0:
                continue
            elif len(s.individuals) == 1:
                misc += s.individuals
            else:
                l.append(s)

        if len(misc) > 0:
            aux = Specie(misc[0])
            aux.individuals = misc
            l.append(aux)
        self.species = l
        pass
