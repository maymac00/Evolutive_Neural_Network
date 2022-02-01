from NN import NN, ENN
from EvolutiveLearning import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

"""
net = ENN([2, 3, 5, 3,  1], activation=['relu', 'relu', 'relu', 'relu'])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [0]])

net.fitness(X, y)
"""


def normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def denormalize(df, f):
    return


pd.testing.N = 10
pd.testing.K = 10
df = pd.util.testing.makeDataFrame().to_numpy()

y = df[:, -1]
x = df[:, :-1]

NEAT.x = x
NEAT.y = y

gens = 30

p = Population(20, 3, 1)
for r in range(gens):
    p.nextGen()

plt.plot(range(gens), p.gen_fitness)
plt.show()

pass
