from NN import NN, ENN
from EvolutiveLearning import Generation, NEAT
import numpy as np

"""
net = ENN([2, 3, 5, 3,  1], activation=['relu', 'relu', 'relu', 'relu'])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [0]])

net.fitness(X, y)
"""

# g = Generation(5, 5, 3)
n = NEAT(3, 1)

pass
