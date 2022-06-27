from numpy.random import rand

from Evolutive_Neural_Network.NEAT import NEAT


class ConnectionGene:
    def __init__(self, i, o, weight=rand() * 4 - 2, enable=True, ):
        self.inp = i
        self.out = o
        self.w = weight
        self.enable = enable
        self.innovation = -1
        NEAT.check_innovation(self)

    def __eq__(self, other):
        if not isinstance(other, ConnectionGene):
            NotImplemented
        return self.inp == other.inp and self.out == other.out

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.inp, self.out))
