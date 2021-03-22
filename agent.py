import torch
import torch.nn
import random
import numpy as np
import yaml
import argparse
from collections import deque
from game.game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from utils import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, args, model):
        self.parameters_file = args.parameters_file
        self.args = args
        self.parameters = yaml.load(open(self.parameters_file, 'r'), Loader=yaml.FullLoader)
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=self.parameters["max_memory"]) # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=self.parameters["lr"], gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger is straight if
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger is right if
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger is left if
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) # converting to 0 or 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def predict(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state_tensor) # moves depending on the model
        move = torch.argmax(prediction).item()
        return move

    def get_action(self, state):
        move = 0
        final_move = [0,0,0]
        # random moves: tradeoff exploration / exploitation
        if self.args.use_trained == True:
            move = self.predict(state)
        else:
            self.epsilon = 100 - self.n_games
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0,2)
            else:
                move = self.predict(state)

        final_move[move] = 1

        return final_move