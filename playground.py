import numpy as np

from Population import Population
from matplotlib import pyplot as plt
import gym

import time
from NEAT import NEAT

"""
This script is used to test the performance of the NEAT algorithm.
In this script we define some enviroments coming from the gym library and we test the performance of the algorithm on them.


"""


def pendulum(ind, render=False, seed=-1):
    rew = 0
    NEAT.activation_function = NEAT.sigmoid_mod
    NEAT.inner_activation_function = NEAT.linear

    NEAT.game = "Pendulum-v1"

    env = gym.make("Pendulum-v1")
    NEAT.max_rwd = 998
    NEAT.step = 2.5
    action = [0]

    NEAT.dropoff = 30

    if NEAT.seed != -1:
        env.reset(seed=NEAT.seed)
    else:
        env.reset()

    ind.log = []
    n_epoch = 1000
    for _ in range(n_epoch):
        if _ == 0 and render:
            env.render()
            time.sleep(0.51)
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        observation[0] = NEAT.normalize_1_1(observation[0], min=-1, max=1) * 2
        observation[1] = NEAT.normalize_1_1(observation[1], min=-1, max=1) * 2
        observation[2] = NEAT.normalize_1_1(observation[2], min=-8, max=8) * 2

        res = ind.process(observation)
        res = np.array(res)

        ind.log.append(res[0])
        action = res

        rew += reward
        if done:
            break
    env.close()
    max_reward = 0
    min_reward = -16.2736044 * n_epoch
    rew = (rew - min_reward) / (max_reward - min_reward) * 1000
    if render:
        time.sleep(1)
        print(rew)

    return rew


def cartpole(ind, render=False):
    rew = 0
    env = gym.make("CartPole-v1")
    NEAT.game = "CartPole-v1"
    NEAT.max_rwd = 500
    NEAT.opt = "max"

    action = env.action_space.sample()

    if NEAT.seed != -1:
        env.reset(seed=NEAT.seed)
    else:
        env.reset()

    for _ in range(1000):
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        res = ind.process(observation)

        # Cartpole-v1 #
        action = res.index(max(res))

        rew += reward
        if done:
            break
    env.close()
    if render:
        time.sleep(1)
        print(rew)
    return rew


def cartpole_2in(ind, render=False):
    rew = 0
    env = gym.make("CartPole-v1")
    NEAT.game = "CartPole-v1-2in"
    NEAT.max_rwd = 500
    NEAT.opt = "max"


    NEAT.activation_function = NEAT.relu
    NEAT.inner_activation_function = NEAT.linear

    action = env.action_space.sample()

    if NEAT.seed != -1:
        env.reset(seed=NEAT.seed)
    else:
        env.reset()

    for _ in range(1000):
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        res = ind.process(observation[:1:-1])

        # Cartpole-v1 #
        action = res.index(max(res))

        rew += reward
        if done:
            break
    env.close()
    if render:
        time.sleep(1)
        print(rew)
    return rew


def cartpole_3in(ind, render=False):
    rew = 0
    env = gym.make("CartPole-v1")
    NEAT.game = "CartPole-v1-3in"
    NEAT.max_rwd = 500
    NEAT.opt = "max"
    NEAT.step = 1.5

    NEAT.activation_function = NEAT.sigmoid_mod
    NEAT.inner_activation_function = NEAT.linear

    action = env.action_space.sample()

    if NEAT.seed != -1:
        env.reset(seed=NEAT.seed)
    else:
        env.reset()

    for _ in range(1000):
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        res = ind.process(observation[:-1])

        # Cartpole-v1 #
        action = res.index(max(res))

        rew += reward
        if done:
            break
    env.close()
    if render:
        time.sleep(1)
        print(rew)
    return rew


def cartpole_ext(ind, render=False):
    rew = 150000
    env = gym.make("CartPole-BT-v0")
    NEAT.max_rwd = np.Inf
    NEAT.activation_function = NEAT.sigmoid_mod
    NEAT.inner_activation_function = NEAT.sigmoid_mod
    action = env.action_space.sample()

    if NEAT.seed != -1:
        env.reset(seed=NEAT.seed)
    else:
        env.reset()

    for _ in range(1000):
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        res = ind.process(observation)
        res = np.array(res)

        action = np.multiply(res, 100)
        # print(action)
        rew += reward
        if done:
            break
    env.close()
    if render:
        time.sleep(1)
        print(rew)
    return rew / 100


if __name__ == "__main__":

    # Environment
    NEAT.fitness = pendulum

    # Number of input and output nodes (problem constrained)
    n_in = 3
    n_out = 1

    # Fixed seed (-1 for random)
    NEAT.seed = 0

    # Run times (should be 1 for seeded runs, and >1 for random runs)
    NEAT.reps = 1

    # Population size
    n_individuals = 100

    # Number of runs
    trys = 3

    # Number of generations
    gens = 20

    # Interval of rendering the best individual (0 to disable)
    show_best = 25

    # Run the algorithm
    trys_bests = []
    for t in range(trys):
        p = Population(n_individuals, n_in, n_out)

        for r in range(gens):
            if r != 0:
                if r % show_best == 0:
                    NEAT.fitness(p.best, True)
            p.nextGen()

        NEAT.fitness(p.best, True)
        trys_bests.append(p.bests)
        NEAT.distance_thld = NEAT.def_distance_thld

    for i in range(len(trys_bests)):
        trys_bests[i] = trys_bests[i][1:]
    trys_bests = np.array(trys_bests)
    plt.figure(2)
    ax = plt.gca()
    ax.set_ylim([trys_bests.min(), 1000])
    for i in range(len(trys_bests)):
        plt.plot(trys_bests[i], label="try: " + str(i))
    plt.show()

    pass
