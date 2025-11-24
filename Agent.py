import torch
import numpy as np
import random
from main import AvoidBallEnv
from collections import deque

from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3


class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9  # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(44, 256, 5)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if max memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(10, 100 - self.n_games * 0.5)
        final_move = 0
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 4)
            final_move = move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = move
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = AvoidBallEnv()
    frame_count = 0

    while True:
        state_old = game._get_state()

        final_move = agent.get_action(state_old)

        state_new, base_reward, done, info = game.step(final_move)

        frame_count += 1

        min_distance = float('inf')
        for rect in game.rectangles:
            dx = rect.x - game.ball_rect.centerx
            dy = rect.y - game.ball_rect.centery
            dist = (dx * dx + dy * dy) ** 0.5
            min_distance = min(min_distance, dist)

        distance_reward = min_distance / 200.0
        speed = (game.ball_velocity[0] ** 2 + game.ball_velocity[1] ** 2) ** 0.5
        movement_reward = speed / 20.0

        if not done:
            reward = base_reward + distance_reward * 0.5 + movement_reward * 0.2
        else:
            reward = base_reward

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        game.render()

        if done:
            state_old = game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            current_score = frame_count

            if current_score > record:
                record = current_score
                agent.model.save()

            print(f'Game {agent.n_games} | Frames Survived: {frame_count} | Record: {record}')
            frame_count = 0


if __name__ == '__main__':
    train()