from NN import NN, ENN
from EvolutiveLearning import *
import numpy as np
from matplotlib import pyplot as plt

# np.random.seed(1)

gens = 150
trys = 1

for t in range(trys):
    p = Population(20, 3, 1)
    for r in range(gens):
        print(p)
        # NEAT.fitness(p.best, True)
        p.nextGen()

    plt.figure(1)
    plt.title("try: " + str(t))
    print("")
    print("try: " + str(t))
    print("S, I: " + str(len(p.species)) + " " + str(len(p.species[0].individuals)))
    print("")
    plt.plot(range(len(p.bests)), p.bests, )
    plt.plot(range(len(p.means)), p.means, )
plt.show()

pass
