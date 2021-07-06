from NN import NN
from EvolutiveLearning import Generation
import numpy as np

net = NN([2, 3, 5, 3,  1], activation=['relu', 'relu', 'relu', 'relu'])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [0]])

g = Generation(5, net)
pass
