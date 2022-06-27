from playground import pendulum, cartpole, cartpole_2in, cartpole_3in, cartpole_ext
from IndividualFactory import IndividualFactory
from NEAT import NEAT


# Pendulum #
filename = "Mcclurg_999.9" + ".json"
# filename = "Null_959.901" + ".json"
# filename = "Kato_921.175" + ".json"
# filename = "Burton_973.388" + ".json"


path = "Pendulum-v1/" + filename

ind = IndividualFactory.buildFromFile(path)

NEAT.fitness = pendulum
NEAT.seed = 1
NEAT.fitness(ind, True)





