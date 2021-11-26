import torch
import random
import numpy as np
from collections import deque
from drone_environment import DroneEnvironment, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
BLOCK_SIZE = 20
LR = 0.001

class Agent():
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(12, 256, 4) # TODO
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # TODO
        # TODO: model, trainer

    def get_state(self, game):
        drone_points = game.drone
        collision_l = any([game.collides(Point(pt.x - BLOCK_SIZE, pt.y)) for pt in drone_points])
        collision_r = any([game.collides(Point(pt.x + BLOCK_SIZE, pt.y)) for pt in drone_points])
        collision_u = any([game.collides(Point(pt.x, pt.y - BLOCK_SIZE)) for pt in drone_points])
        collision_d = any([game.collides(Point(pt.x, pt.y + BLOCK_SIZE)) for pt in drone_points])

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_l and collision_l) or
            (dir_r and collision_r) or
            (dir_u and collision_u) or
            (dir_d and collision_d),

            # Danger right
            (dir_u and collision_r) or
            (dir_r and collision_d) or
            (dir_d and collision_l) or
            (dir_l and collision_u),

            # Danger left
            (dir_d and collision_r) or
            (dir_r and collision_u) or
            (dir_u and collision_l) or
            (dir_l and collision_d),

            # Danger behind
            (dir_l and collision_r) or
            (dir_r and collision_l) or
            (dir_u and collision_d) or
            (dir_d and collision_u),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Object location
            game.object.x < game.gripper.x, # Object left
            game.object.x > game.gripper.x, # Object right
            game.object.y < game.gripper.y, # Object top
            game.object.y > game.gripper.y, # Object bottom
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 500 - self.num_games
        final_move = [0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state_zero = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_zero)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = DroneEnvironment()
    # while True:
    while agent.num_games < 10000:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.num_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()