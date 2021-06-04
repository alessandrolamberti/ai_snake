import torch
import torch.nn
import random
import numpy as np
import argparse
import yaml
from collections import deque
from game.game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from agent import Agent
from utils import plot


def main():

    parameters = yaml.load(open(args.parameters_file, 'r'),
            Loader=yaml.FullLoader)
    
    model = Linear_QNet(11, 256, 3)

    if args.use_trained == True:
        model.load_state_dict(torch.load(parameters["model_path"]))

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(args, model)
    game = SnakeGameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                if args.save_model == True:
                    agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--parameters_file", 
        default = "parameters.yml", type = str)

    parser.add_argument(
        "--use_trained", default=False, type = bool)
    
    parser.add_argument(
        "--save_model", default=False, type = bool)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main()