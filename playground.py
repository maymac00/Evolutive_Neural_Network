import numpy as np

from Population import *
from matplotlib import pyplot as plt
import gym

from snake import snake_game

import time
import NEAT
# np.random.seed(1)
seed = int(rand() * 10)


def pendulum(ind, render=False):
    rew = 1800
    NEAT.NEAT.activation_function = NEAT.NEAT.linear
    NEAT.NEAT.inner_activation_function = NEAT.NEAT.relu

    env = gym.make("Pendulum-v0")
    NEAT.NEAT.max_rwd = 2000
    NEAT.NEAT.blood_rate = 1

    # env.seed(1)
    action = [0]
    NEAT.NEAT.reps = 5
    env.reset()
    ind.log = []
    for _ in range(1000):
        if render:
            time.sleep(0.01)
            env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        observation[0] = NEAT.NEAT.normalize(observation[0], min=-1, max=1)
        observation[1] = NEAT.NEAT.normalize(observation[1], min=-1, max=1)
        observation[2] = NEAT.NEAT.normalize(observation[2], min=-8, max=8)

        res = ind.process(observation)
        res = np.array(res)

        ind.log.append(res[0])
        action = res

        rew += reward
        if done:
            break
    env.close()
    if render:
        time.sleep(1)
        print(rew)
    return rew


def cartpole(ind, render=False):
    rew = 0
    env = gym.make("CartPole-v1")
    NEAT.NEAT.max_rwd = 500
    NEAT.NEAT.opt = "max"
    NEAT.NEAT.reps = 4
    action = env.action_space.sample()
    # env.seed(1)
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
    NEAT.NEAT.max_rwd = 500
    NEAT.NEAT.opt = "max"
    NEAT.NEAT.reps = 4
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
    NEAT.NEAT.max_rwd = 500
    NEAT.NEAT.opt = "max"
    NEAT.NEAT.reps = 4
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

        observation[0] = NEAT.NEAT.normalize(observation[0], min=-3.5, max=3.5)
        observation[1] = NEAT.NEAT.normalize(observation[0], min=-3.3, max=3.3)
        observation[2] = NEAT.NEAT.normalize(observation[0], min=-3.5, max=3.5)
        observation[3] = NEAT.NEAT.normalize(observation[0], min=-3.5, max=3.5)
        observation[4] = NEAT.NEAT.normalize(observation[0], min=-3.5, max=3.5)
        observation[5] = NEAT.NEAT.normalize(observation[0], min=-3.5, max=3.5)
        observation[6] = NEAT.NEAT.normalize(observation[0], min=-3.5, max=3.5)
        observation[7] = NEAT.NEAT.normalize(observation[0], min=-4.5, max=4.5)

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


NEAT.NEAT.fitness = pendulum

NEAT.NEAT.max_rwd = np.inf
NEAT.NEAT.adaptation = 1
gens = 600
trys = 1
NEAT.NEAT.activation_function = NEAT.NEAT.sigmoid_mod
NEAT.NEAT.inner_activation_function = NEAT.NEAT.relu

for t in range(trys):
    p = Population(150, 3, 1)

    for r in range(gens):
        if r % 3 == 0:
            NEAT.NEAT.fitness(p.best, True)
        p.nextGen()

    NEAT.NEAT.fitness(p.best, True)
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
