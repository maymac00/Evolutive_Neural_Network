from numpy.random import rand

from NEAT import NEAT

"""
ConnectionGene class

    This class is used to represent a connection between two nodes.
    It contains the weight of the connection, the innovation number and
    the enabled status
"""


class ConnectionGene:
    def __init__(self, i, o, weight=rand() * 4 - 2, enable=True, innovation=-1):
        self.inp = i
        self.out = o
        self.w = weight
        self.enable = enable
        self.innovation = innovation
        if self.innovation == -1:
            NEAT.check_innovation(self)

    def __eq__(self, other):
        if not isinstance(other, ConnectionGene):
            NotImplemented
        return self.inp == other.inp and self.out == other.out

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.inp, self.out))

    def toJson(self):
        s = "{\"inp\":" + str(self.inp) + ",\"out\": " + str(self.out) + ",\"w\": " + str(self.w) + ",\"enable\": " + (
            "1" if self.enable else "0") + ",\"innovation\": " + str(self.innovation) + "}"
        return s
