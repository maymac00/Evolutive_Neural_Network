# importing libraries
import copy
import math

import numpy as np
import pygame
import time
import random

from Evolutive_Neural_Network.Individual import NEAT


def snake_game(ind, render=False):
    NEAT.distance_thld = 50
    NEAT.max_rwd = np.inf

    snake_speed = 3000

    if render:
        snake_speed = 30
    # Window size
    window_x = 720
    window_y = 480
    max_dist = np.ceil(np.sqrt(window_x * window_x + window_y * window_y))
    # defining colors
    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)

    # Initialising pygame
    pygame.init()

    # Initialise game window
    pygame.display.set_caption('GeeksforGeeks Snakes')
    if render:
        game_window = pygame.display.set_mode((window_x, window_y))

    # FPS (frames per second) controller
    fps = pygame.time.Clock()

    # defining snake default position
    snake_position = np.array([100, 50])

    # defining first 4 blocks of snake body
    snake_body = [[100, 50],
                  [90, 50],
                  [80, 50],
                  [70, 50]
                  ]
    # fruit position
    fruit_position = np.array([random.randrange(1, (window_x // 10)) * 10,
                               random.randrange(1, (window_y // 10)) * 10])

    fruit_spawn = True

    # setting default snake direction towards
    # right
    direction = 'RIGHT'
    change_to = direction

    # initial score
    score = 1

    # displaying Score function
    def show_score(choice, color, font, size):
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)

        # create the display surface object
        # score_surface
        score_surface = score_font.render('Score : ' + str(score), True, color)

        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()

        # displaying text
        if render:
            game_window.blit(score_surface, score_rect)

    def get_inps():
        inps = np.full(26, max_dist)
        init = snake_position
        # 1
        vecs = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        """
        if direction == 'RIGHT':
            theta = 0
        elif direction == 'UP':
            theta = math.pi/2
        elif direction == 'LEFT':
            theta = math.pi
        else:
            theta = -math.pi/2

        vec2 = []
        for v in vecs:
            x = np.round(v[0] * math.cos(theta) - v[1]*math.sin(theta))
            y = np.round(v[0] * math.sin(theta) + v[1]*math.cos(theta))
            vec2.append([x, y])"""
        for i, vec in enumerate(vecs):
            cont = 0
            it = [-1, -1]
            while it[0] != 0 and it[0] != window_x and it[1] != 0 and it[1] != window_y:
                it = [init[0] + cont * vec[0], init[1] + cont * vec[1]]
                if it == fruit_position:
                    inps[i * 3 + 0] = abs(init[0] - it[0])
                if it in snake_body and it != snake_position:
                    inps[i * 3 + 1] = abs(init[0] - it[0])
                cont += 10
            inps[i * 3 + 2] = cont - 10

        inps[24] = fruit_position[0] - snake_position[0]
        inps[25] = fruit_position[1] - snake_position[1]
        inps[:24] = NEAT.normalize(inps[:24], min=10, max=max_dist)
        inps[24] = NEAT.normalize(inps[24], min=-max_dist, max=max_dist)
        inps[25] = NEAT.normalize(inps[25], min=-max_dist, max=max_dist)
        return inps

    def get_inps2():
        inps = np.zeros(12)
        init = snake_position
        vecs = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        for i, vec in enumerate(vecs):
            cont = 0
            it = [-1, -1]
            while it[0] != 0 and it[0] != window_x and it[1] != 0 and it[1] != window_y:
                it = np.array([init[0] + cont * vec[0], init[1] + cont * vec[1]])
                if all(it == fruit_position):
                    inps[i] = 1
                    break
                cont += 10

        for i, v in enumerate(vecs):
            newpos = snake_position + 10 * np.array(v)
            if newpos[0] != 0 or newpos[0] != window_x or newpos[1] != 0 or newpos[
                1] != window_y or newpos in snake_body:
                inps[4 + i] = 1
        if direction == 'RIGHT':
            inps[9] = 1
        elif direction == 'UP':
            inps[8] = 1
        elif direction == 'LEFT':
            inps[11] = 1
        else:
            inps[10] = 1
        return inps

    # Main Function
    i = 0
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    max_its = 500

    def dist():
        return np.sqrt(np.sum(np.square(fruit_position-snake_position)))

    prev_dist = dist()
    while i < max_its:

        if ind is not None:
            inps = get_inps2()
            # handling key events
            res = ind.process(inps)
            action = res.index(max(res))
            change_to = actions[action]
        else:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        change_to = 'RIGHT'

        # If two keys pressed simultaneously
        # we don't want snake to move into two
        # directions simultaneously
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Moving the snake
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10

        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        snake_body.insert(0, list(snake_position))
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 10
            fruit_spawn = False
        else:
            snake_body.pop()

        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                              random.randrange(1, (window_y // 10)) * 10]

        fruit_spawn = True
        if render:
            game_window.fill(black)

            for pos in snake_body:
                pygame.draw.rect(game_window, green,
                                 pygame.Rect(pos[0], pos[1], 10, 10))
            pygame.draw.rect(game_window, white, pygame.Rect(
                fruit_position[0], fruit_position[1], 10, 10))

        # Game Over conditions
        if snake_position[0] < 0 or snake_position[0] > window_x - 10:
            if render:
                time.sleep(2)
                pygame.quit()
            return max(score, 0)
        if snake_position[1] < 0 or snake_position[1] > window_y - 10:
            if render:
                time.sleep(2)
                pygame.quit()
            return max(score, 0)

        # Touching the snake body
        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                if render:
                    time.sleep(2)
                    pygame.quit()
                return max(score, 0)

        # Refresh game screen
        if render:
            pygame.display.update()

        # Frame Per Second /Refresh Rate
        fps.tick(snake_speed)
        i += 1

        d = dist()
        if d < prev_dist:
            score += 1
        else:
            score -= 1
        prev_dist = d
    if render:
        time.sleep(2)
        pygame.quit()
    return max(score, 0)
# snake(None, True)
