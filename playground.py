import numpy as np

from Population import *
from matplotlib import pyplot as plt
import gym

from snake import snake_game

import time
import NEAT
# np.random.seed(1)
seed = int(rand() * 10)


def pendulum(ind, render=False, seed=1):
    rew = 0
    NEAT.NEAT.activation_function = NEAT.NEAT.sigmoid_mod
    NEAT.NEAT.inner_activation_function = NEAT.NEAT.linear

    NEAT.NEAT.game = "Pendulum-v1"

    env = gym.make("Pendulum-v1")
    NEAT.NEAT.max_rwd = 998
    NEAT.NEAT.blood_rate = 1
    NEAT.NEAT.step = 2.5
    action = [0]

    NEAT.NEAT.reps = 4
    NEAT.NEAT.dropoff = 30

    if NEAT.NEAT.seed != 0:
        env.reset(seed=NEAT.NEAT.seed)
    else:
        env.reset()

    ind.log = []
    n_epoch = 1000
    for _ in range(n_epoch):
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        observation[0] = NEAT.NEAT.normalize_1_1(observation[0], min=-1, max=1) * 2
        observation[1] = NEAT.NEAT.normalize_1_1(observation[1], min=-1, max=1) * 2
        observation[2] = NEAT.NEAT.normalize_1_1(observation[2], min=-8, max=8) * 2

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
    NEAT.NEAT.game = "CartPole-v1"
    NEAT.NEAT.max_rwd = 500
    NEAT.NEAT.opt = "max"
    NEAT.NEAT.reps = 4
    action = env.action_space.sample()

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
    NEAT.NEAT.game = "CartPole-v1-2in"
    NEAT.NEAT.max_rwd = 500
    NEAT.NEAT.opt = "max"
    NEAT.NEAT.reps = 4

    NEAT.NEAT.activation_function = NEAT.NEAT.relu
    NEAT.NEAT.inner_activation_function = NEAT.NEAT.linear

    action = env.action_space.sample()
    # env.seed(1)
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
    NEAT.NEAT.game = "CartPole-v1-3in"
    NEAT.NEAT.max_rwd = 500
    NEAT.NEAT.opt = "max"
    NEAT.NEAT.reps = 4
    NEAT.NEAT.step = 1.5

    NEAT.NEAT.activation_function = NEAT.NEAT.sigmoid_mod
    NEAT.NEAT.inner_activation_function = NEAT.NEAT.linear

    action = env.action_space.sample()
    # env.seed(1)
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
    NEAT.NEAT.max_rwd = np.Inf
    NEAT.NEAT.activation_function = NEAT.NEAT.sigmoid_mod
    NEAT.NEAT.inner_activation_function = NEAT.NEAT.sigmoid_mod
    action = env.action_space.sample()
    NEAT.NEAT.reps = 4

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


def lunarlander(ind, render=False):
    rew = 131
    env = gym.make("LunarLander-v2")
    NEAT.NEAT.max_rwd = 1400
    NEAT.NEAT.opt = "max"
    NEAT.NEAT.reps = 1
    NEAT.NEAT.step = 4
    action = env.action_space.sample()
    env.seed(1)
    env.reset()

    for _ in range(1000):
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        observation[0] = NEAT.NEAT.normalize01(observation[0], min=-3.5, max=3.5)
        observation[1] = NEAT.NEAT.normalize01(observation[0], min=-3.3, max=3.3)
        observation[2] = NEAT.NEAT.normalize01(observation[0], min=-3.5, max=3.5)
        observation[3] = NEAT.NEAT.normalize01(observation[0], min=-3.5, max=3.5)
        observation[4] = NEAT.NEAT.normalize01(observation[0], min=-3.5, max=3.5)
        observation[5] = NEAT.NEAT.normalize01(observation[0], min=-3.5, max=3.5)
        observation[6] = NEAT.NEAT.normalize01(observation[0], min=-3.5, max=3.5)
        observation[7] = NEAT.NEAT.normalize01(observation[0], min=-4.5, max=4.5)

        res = ind.process(observation)

        action = res.index(max(res))

        rew += reward
        if done:
            break
    env.close()
    if render:
        time.sleep(1)
        print(rew)
    return max(rew, 0)


if __name__ == "__main__":
    NEAT.NEAT.fitness = pendulum

    gens = 200
    trys = 3
    trys_bests = []
    for t in range(trys):
        p = Population(100, 3, 1)

        for r in range(gens):
            if r % 25 == 1:
                NEAT.NEAT.fitness(p.best, True)
            p.nextGen()

        NEAT.NEAT.fitness(p.best, True)

        print("")
        print("try: " + str(t))
        print("S, I: " + str(len(p.species)) + " " + str(len(p.species[0].individuals)))
        print("")
        trys_bests.append(p.bests)
        NEAT.NEAT.distance_thld = NEAT.NEAT.def_distance_thld


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
