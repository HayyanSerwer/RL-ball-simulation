import torch
import numpy as np
import random
from main import AvoidBallEnv, FloatingRect, score
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #to control the randomness
        self.gamma = 0 # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        # model, trainer

    def getstate(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = AvoidBallEnv()
    while True:
        state_old = agent.getstate(game)

        # get move based on this current state

        final_move = agent.get_action(state_old)

        # perform the move and then get the new state
        state, reward, done, info = game.step(final_move)

        state_new = agent.getstate(game)

        #Train the short memory (only for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)


if __name__ == '__main__':
    train()



