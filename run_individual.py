from playground import pendulum, cartpole, cartpole_2in, cartpole_3in, cartpole_ext
from IndividualFactory import IndividualFactory
from NEAT import NEAT
filename = "Lawrence_999.778" + ".json"


path = "Pendulum-v1/" + filename

ind = IndividualFactory.buildFromFile(path)

NEAT.fitness = pendulum
NEAT.seed = 1
NEAT.fitness(ind, True)





