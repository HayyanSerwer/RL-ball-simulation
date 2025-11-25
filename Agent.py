import torch
import numpy as np
import random
from main import AvoidBallEnv, CIRCLE_RADIUS
from collections import deque
import math
from model import Linear_QNet, QTrainer



MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-4

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPS_START  # Use EPS_START instead of 0
        self.gamma = 0.99  # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(16, 512, 5)
        self.target_model = Linear_QNet(16, 512, 5)
        self.target_model.load_state_dict(self.model.state_dict())
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

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
        # FIX 5: Use exponential decay for epsilon
        self.epsilon = max(EPS_END, EPS_START * math.pow(EPS_DECAY, self.n_games))


        final_move = 0
        if random.random() < self.epsilon:
            # Exploration: Take a random action
            final_move = random.randint(0, 4)
        else:
            # Exploitation: Get the best action from the model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            final_move = torch.argmax(prediction).item()

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

        state_new, reward, done, info = game.step(final_move)

        frame_count += 1

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        # game.render()
        if done:
            state_old = game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.trainer.update_target()

            current_score = frame_count

            if current_score > record:
                record = current_score
                agent.model.save()

            print(f'Game {agent.n_games} | Epsilon: {agent.epsilon:.3f} | Frames Survived: {frame_count} | Record: {record}')
            frame_count = 0


if __name__ == '__main__':
    train()