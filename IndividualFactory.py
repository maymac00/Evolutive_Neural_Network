import numpy as np
from Individual import Individual


class IndividualFactory:

    @staticmethod
    def buildIndividual(inp, out, genome):
        ind = Individual(inp, out)
        for gen in genome:
            if gen.out not in (ind.out + ind.hidden):
                ind.matrix = np.hstack([ind.matrix, np.atleast_2d([None for j in range(ind.n_neurons)]).T])
                ind.matrix = np.vstack([ind.matrix, [None for j in range(ind.n_neurons + 1)]])
                ind.hidden.append(gen.out)
                ind.n_neurons += 1
            n = ind.add_connection(gen.inp, gen.out, force=True)
            ind.genome[n].w = gen.w
            ind.genome[n].enable = gen.enable
            ind.genome[n].innovation = gen.innovation
        return ind

    @staticmethod
    def buildRandomLike(param):
        n = len(param.genome.keys())
        ind = Individual(len(param.inp), len(param.out))
        while len(ind.genome.keys()) < n:
            ind.force_mutate()
        return ind
        pass
