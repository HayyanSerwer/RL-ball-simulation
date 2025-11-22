import torch
import numpy as np
import random
from main import AvoidBallEnv, FloatingRect, score, WIDTH, HEIGHT, CIRCLE_RADIUS
from collections import deque
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #to control the randomness
        self.gamma = 0 # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = None  # TODO
        self.trainer = None  # TODO
        # model, trainer

    def getstate(self, game):
        # --- Ball info ---
        bx = self.ball_rect.centerx
        by = self.ball_rect.centery
        bvx = self.ball_velocity[0]
        bvy = self.ball_velocity[1]

        state = [
            bx / WIDTH,
            by / HEIGHT,
            bvx / 10,
            bvy / 10,
        ]

        # --- Collect (distance, rect) for sorting ---
        rect_info = []
        for rect in self.rectangles:
            dx = rect.x - bx
            dy = rect.y - by
            dist = math.sqrt(dx * dx + dy * dy)
            rect_info.append((dist, rect))

        # Sort by nearest
        rect_info.sort(key=lambda x: x[0])

        # --- Add the 10 rectangles (they are already 10, but still sorted) ---
        MAX_SPEED = 5  # max rect velocity for normalization

        for dist, rect in rect_info:
            # Normalize distances relative to circle radius
            dx = (rect.x - bx) / CIRCLE_RADIUS
            dy = (rect.y - by) / CIRCLE_RADIUS

            # Normalize velocities
            dvx = rect.vx / MAX_SPEED
            dvy = rect.vy / MAX_SPEED

            state.extend([dx, dy, dvx, dvy])

        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if max memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples,
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, done = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # This is the tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move = 0
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,3)
            final_move = move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()

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

        # remmeber
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory (also known as experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save later when we have the model

            print('Game', agent.n_games, 'Score', score, 'Record', record)




if __name__ == '__main__':
    train()



