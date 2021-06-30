from NN import NN
import numpy as np

net = NN([2, 3, 5, 3,  1], activation=['relu', 'relu', 'relu', 'relu'])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [0]])

for i in range(4):
    print(net.topology)

    err, it = net.train(X, y)

    print(err, "its: ", it)

    net.reset()
    net.addNeuron()
