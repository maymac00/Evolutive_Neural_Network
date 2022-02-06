from NN import NN, ENN
from EvolutiveLearning import *
import numpy as np
from matplotlib import pyplot as plt

"""
net = ENN([2, 3, 5, 3,  1], activation=['relu', 'relu', 'relu', 'relu'])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [0]])

net.fitness(X, y)
"""

np.random.seed(1)

gens = 15
trys = 1

for t in range(trys):
    p = Population(10, 4, 2)
    p.nextGen()
    for r in range(gens):
        print(str(len(p.species)) + " " + str(len(p.species[0].individuals))+ "--> " + str(p.means[-1]))
        # NEAT.fitness(p.current_bests_inds[0], True)
        p.nextGen()

    plt.figure(1)
    plt.title("try: " + str(t))
    print("")
    print("try: " + str(t))
    print("S, I: " + str(len(p.species)) + " " + str(len(p.species[0].individuals)))
    print("")
    plt.plot(range(gens+1), p.bests, )
    plt.plot(range(gens+1), np.multiply(p.means, 5), )
plt.show()

pass
