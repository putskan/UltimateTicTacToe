import os
from pathlib import Path
from typing import List, Tuple
from itertools import permutations
from collections import Counter
from logging import Logger
import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from pettingzoo import AECEnv

from agents.agent import Agent
from agents.choose_first_action_agent import ChooseFirstActionAgent
from agents.hierarchical_agent import HierarchicalAgent
from agents.random_agent import RandomAgent
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm

from agents.unbeatable_classic_ttt_agent.unbeatable_classic_ttt_agent import UnbeatableClassicTTTAgent
from environments import ultimate_ttt
from utils.utils import get_action_mask
from utils.logger import get_logger

K = 32
s = 400
INITIAL_ELO_RATE = 1200


class AgentEvaluator:
    def __init__(self, agents: List[Agent], env: AECEnv, logger: Logger = None, n_rounds: int = 1000):
        n_agents = len(agents)
        if logger is None:
            logger = get_logger("evaluate_agents", log_dir_name=None, log_to_console=True)

        self.agents = agents
        self.n_rounds = n_rounds
        self.env = env
        self.logger = logger
        self.wins = np.zeros((n_agents, n_agents), dtype=int)
        self.players_elo_rating = {agent: INITIAL_ELO_RATE for agent in agents}
        self.player_total_games = 2 * (len(self.agents) - 1) * self.n_rounds

    def play_single_game(self, players: Tuple[Agent, Agent]) -> Tuple[float, float]:
        """
        play n_games between the players and print the results
        :param env: env to play in
        :param players: 2 players to play the game
        :return: results of the games - score for each player: win - 1, draw - 0.5, lose - 0
        """
        cumulative_rewards = [0] * len(players)
        curr_player_idx = 0
        self.env.reset()
        for curr_agent_str in self.env.agent_iter():
            curr_player = players[curr_player_idx]
            observation, reward, termination, truncation, info = self.env.last()
            cumulative_rewards[curr_player_idx] += reward
            action_mask = get_action_mask(observation, info)
            done = termination or truncation
            if done:
                action = None
            else:
                action = curr_player.play(self.env, observation, curr_player_idx, curr_agent_str, action_mask, info)
            self.env.step(action)
            curr_player_idx = (curr_player_idx + 1) % len(players)

        # scores: win - 1, draw - 0.5, lose - 0
        score1 = (cumulative_rewards[0] + 1) / 2
        score2 = (cumulative_rewards[1] + 1) / 2
        self.env.close()
        return score1, score2

    def log_final_results(self, title: str, final_results: List[Tuple[str, float]], use_percentage: bool = False) -> None:
        """
        Logs the final results of the evaluation
        :param logger: logger to log results
        :param title: title of the results
        :param final_results: list of tuples containing the agent name and the winning percentage, sorted by winning percentage
        """
        self.logger.info("************************************")
        self.logger.info(f"Final {title} Results:")
        for agent_name, winning_percentage in final_results:
            msg = f"{agent_name}: {winning_percentage:.2f}"
            if use_percentage:
                msg = f"{agent_name}: {winning_percentage:.2%}"
            self.logger.info(msg)
        self.logger.info("************************************")
        self.logger.info("")

    @staticmethod
    def calc_new_elo_rating(rating1: float, rating2: float, score1: float, score2: float) -> Tuple[float, float]:
        """
        Calculates the new elo rating for two players
        :param rating1: current elo rating of player 1
        :param rating2: current elo rating of player 2
        :param score1: score of player 1 in the game (1 for win, 0.5 for draw, 0 for loss)
        :param score2: score of player 2 in the game (1 for win, 0.5 for draw, 0 for loss)
        :return: new elo ratings for player 1 and player 2
        """
        expected_score1 = 1 / (1 + 10 ** ((rating2 - rating1) / s))
        expected_score2 = 1 / (1 + 10 ** ((rating1 - rating2) / s))
        new_rating1 = rating1 + K * (score1 - expected_score1)
        new_rating2 = rating2 + K * (score2 - expected_score2)
        return new_rating1, new_rating2

    def evaluate_agents(self) -> None:
        """
        Run games between the agents and assess their performance
        :param env: env to play in
        :param agents: list of agents to evaluate
        :param logger: logger to log results. if None, default logger is created and logs only to console
        :param n_rounds: number of rounds to play.
                        for n_rounds=n and len(agents)=k, we play n * (k * (k-1)) games
        """
        n_agents = len(self.agents)
        all_agent_pairs = list(permutations(range(n_agents), 2))
        for _ in tqdm(range(self.n_rounds)):
            for agent1_idx, agent2_idx in all_agent_pairs:
                agent1, agent2 = self.agents[agent1_idx], self.agents[agent2_idx]
                score1, score2 = self.play_single_game((agent1, agent2))

                # update win count of each player
                self.wins[agent1_idx, agent2_idx] += score1 == 1
                self.wins[agent2_idx, agent1_idx] += score2 == 1

                # update elo ranting of each player
                rate1, rate2 = self.players_elo_rating[agent1], self.players_elo_rating[agent2]
                new_rate1, new_rate2 = self.calc_new_elo_rating(rate1, rate2, score1, score2)
                self.players_elo_rating[agent1] = new_rate1
                self.players_elo_rating[agent2] = new_rate2

        player_draws = self.player_total_games - (self.wins.sum(axis=0) + self.wins.sum(axis=1))
        players_win_percentage = (self.wins.sum(axis=1) + player_draws) / self.player_total_games
        players_win_percentage_dict = {self.agents[i]: players_win_percentage[i] for i in range(n_agents)}
        sorted_player_win_percentage = sorted(players_win_percentage_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_elo_ratings = sorted(self.players_elo_rating.items(), key=lambda item: item[1], reverse=True)

        # log results
        self.logger.info(f'n_rounds={self.n_rounds}, n_agents={len(self.agents)}, '
                    f'total_games={self.n_rounds * len(self.agents) * (len(self.agents) - 1)}')
        self.log_final_results("Winning Percentage", sorted_player_win_percentage, use_percentage=True)
        self.log_final_results("Elo Rating", sorted_elo_ratings)
        self.plot_results()

    @property
    def log_folder(self):
        for handler in self.logger.handlers:
            if hasattr(handler, 'baseFilename'):
                return Path(handler.baseFilename).parent

    def plot_results(self, save_plot: bool = True) -> None:
        # Creating a DataFrame for each matrix for better visualization
        draws = (2 * self.n_rounds) - (self.wins + self.wins.T)
        draws[np.eye(len(draws), dtype=bool)] = 0
        wins_df = pd.DataFrame(self.wins, index=self.agents, columns=self.agents)
        losses_df = pd.DataFrame(self.wins.T, index=self.agents, columns=self.agents)
        draws_df = pd.DataFrame(draws, index=self.agents, columns=self.agents)

        # Visualizing the matrices using heatmaps
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        sns.heatmap(wins_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title('Wins')

        plt.subplot(1, 3, 2)
        sns.heatmap(losses_df, annot=True, fmt="d", cmap="Reds", cbar=False)
        plt.title('Losses')

        plt.subplot(1, 3, 3)
        sns.heatmap(draws_df, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.title('Draws')

        plt.tight_layout()

        if self.log_folder and save_plot:
            plt.savefig(self.log_folder / 'results.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate agents and log results')
    parser.add_argument('--log-to-console', default=1, type=int, choices=(0, 1))
    args = parser.parse_args()

    env = ultimate_ttt.env(render_mode=None, depth=2)  # 'human', 'rgb_array', 'ansi', None
    agents = [HierarchicalAgent(), RandomAgent(), ChooseFirstActionAgent(), RandomAgent("rand"), ChooseFirstActionAgent("choose")]

    os.makedirs('logs', exist_ok=True)
    logger_file_name = f"logs/evaluate_agents_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger = get_logger("evaluate_agents", logger_file_name, log_to_console=args.log_to_console)
    agent_evaluator = AgentEvaluator(agents, env, logger=None, n_rounds=100)
    agent_evaluator.evaluate_agents()
