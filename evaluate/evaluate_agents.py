import os
from pathlib import Path
from typing import List, Tuple, Dict
from itertools import permutations
from logging import Logger
import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from tqdm import tqdm
from pettingzoo import AECEnv

from policy_functions.reinforce_policy import ReinforcePolicy

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from environments import ultimate_ttt
from utils.utils import get_action_mask, load_agent
from utils.logger import get_logger
from agents.agent import Agent
from agents.TimedAgentWrapper import TimedAgentWrapper

matplotlib.use("TkAgg")


K = 32
s = 400
INITIAL_ELO_RATE = 1200


class AgentsEvaluator:
    """
    Class to evaluate agents' performance by playing multiple rounds of games
    """
    def __init__(self, agents: List[Agent], env: AECEnv, logger: Logger = None, n_rounds: int = 1000):
        """
        :param agents: list of agents to evaluate
        :param env: env to play in
        :param logger: logger to log results. if None, default logger is created and logs only to console
        :param n_rounds: number of rounds to play.
                        for n_rounds=n and len(agents)=k, we play n * (k * (k-1)) games
        """
        for agent in agents:
            if hasattr(agent, 'eval'):
                agent.eval()
        n_agents = len(agents)
        if logger is None:
            logger = get_logger("evaluate_agents", log_dir_name=None, log_to_console=True)

        self.agents = [TimedAgentWrapper(agent) for agent in agents]
        self.n_rounds = n_rounds
        self.env = env
        self.logger = logger
        self.wins = np.zeros((n_agents, n_agents), dtype=int)
        self.players_elo_rating = {agent: INITIAL_ELO_RATE for agent in self.agents}
        self.player_total_games = 2 * (len(self.agents) - 1) * self.n_rounds

    def play_single_game(self, players: Tuple[Agent, Agent]) -> Tuple[float, float]:
        """
        play n_games between the players and print the results
        :param players: 2 players to play the game
        :return: results of the games - score for each player: win - 1, draw - 0.5, lose - 0
        """
        cumulative_rewards = [0] * len(players)
        curr_player_idx = 0
        self.env.reset()
        for player in players:
            player.reset()
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
        :param title: title of the results
        :param final_results: list of tuples containing the agent name and the result, sorted results
        :param use_percentage: whether to print the results as a percentage or not
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
        players_win_percentage = (self.wins.sum(axis=1) + 0.5 * player_draws) / self.player_total_games
        players_win_percentage_dict = {self.agents[i]: players_win_percentage[i] for i in range(n_agents)}
        sorted_player_win_percentage = sorted(players_win_percentage_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_elo_ratings = sorted(self.players_elo_rating.items(), key=lambda item: item[1], reverse=True)
        player_avg_time = {str(agent): agent.avg_time for agent in self.agents}
        sorted_times = sorted(player_avg_time.items(), key=lambda item: item[1], reverse=True)

        # log results
        self.logger.info(f'n_rounds={self.n_rounds}, n_agents={len(self.agents)}, '
                    f'total_games={self.n_rounds * len(self.agents) * (len(self.agents) - 1)}')
        self.log_final_results("Winning Percentage", sorted_player_win_percentage, use_percentage=True)
        self.log_final_results("Elo Rating", sorted_elo_ratings)
        self.log_final_results("Average Time per Move (in seconds)", sorted_times)
        self.plot_results()
        self.plot_average_times(player_avg_time)

    @property
    def log_folder(self):
        """
        :return: the folder of the logger, if exists
        """
        for handler in self.logger.handlers:
            if hasattr(handler, 'baseFilename'):
                return Path(handler.baseFilename).parent

    def plot_results(self, save_plot: bool = True) -> None:
        """
        Plots the results as 3 heatmaps of wins, loses and draws.
        Optionally also saves the plots to the logger's folder.
        :param save_plot: whether to save the plot or not
        """
        draws = (2 * self.n_rounds) - (self.wins + self.wins.T)
        draws[np.eye(len(draws), dtype=bool)] = 0
        # Creating a DataFrame for each matrix for better visualization
        wins_df = pd.DataFrame(self.wins, index=self.agents, columns=self.agents)
        losses_df = pd.DataFrame(self.wins.T, index=self.agents, columns=self.agents)
        draws_df = pd.DataFrame(draws, index=self.agents, columns=self.agents)

        # Visualizing the matrices using heatmaps
        plt.figure(figsize=(18, 12))

        plt.subplot(2, 3, 1)
        sns.heatmap(wins_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title('Wins')

        plt.subplot(2, 3, 2)
        sns.heatmap(losses_df, annot=True, fmt="d", cmap="Reds", cbar=False)
        plt.title('Losses')

        plt.subplot(2, 3, 3)
        sns.heatmap(draws_df, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.title('Draws')

        # bar plot for total scores
        total_scores = wins_df.sum(axis=1) + draws_df.sum(axis=1) * 0.5  # Wins + 0.5 * Draws
        plt.subplot(2, 3, 4)
        colors = sns.color_palette("rainbow", len(self.wins))

        total_scores.sort_values().plot(kind='barh', color=colors)
        plt.title('Total Scores by Agent')
        plt.xlabel('Score')
        plt.ylabel('Agent')

        # Elo rating
        elo_ratings = pd.Series(self.players_elo_rating).sort_values()
        plt.subplot(2, 3, 5)
        elo_ratings.plot(kind='barh', color=colors)
        plt.title('Elo Ratings by Agent')
        plt.xlabel('Elo Rating')
        plt.ylabel('Agent')

        plt.suptitle(f'Game Results Between Agents ({2 * self.n_rounds} Games per Pair)', fontsize=14)
        plt.tight_layout()

        if self.log_folder and save_plot:
            plt.savefig(self.log_folder / 'results.png')
        plt.show()

    def plot_average_times(self, agents_times: Dict[str, float], save_plot: bool = True) -> None:
        """
        Plots the results as 3 heatmaps of wins, loses and draws.
        Optionally also saves the plots to the logger's folder.
        :param agents_times: a dictionary of agent names and their average time per move
        :param save_plot: whether to save the plot or not
        """
        agents_times = pd.Series(agents_times).sort_values()
        palette = sns.color_palette("crest")

        plt.figure(figsize=(10, 6))
        bars = plt.bar(agents_times.index, agents_times.values, color=palette)
        plt.xlabel('Agent')
        plt.ylabel('Average Time (seconds)')
        plt.title('Average Time Per Agent')

        # Add the value labels on top of each bar
        for bar, value in zip(bars, agents_times.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X position: center of the bar
                bar.get_height(),
                f'{value:.2f}',
                ha='center',
                va='bottom'
            )

        plt.xlabel('Agent')
        plt.ylabel('Average Time (seconds)')
        plt.title('Average Time Per Agent')
        if self.log_folder and save_plot:
            plt.savefig(self.log_folder / 'avg_time_results.png')
        plt.show()


if __name__ == '__main__':
    from agents.dqn_agent import DQNAgent
    from agents.search_agents.alpha_beta import AlphaBeta
    from agents.search_agents.mcts import MCTSAgent
    from evaluation_functions.ae_winning_possibilities import AEWinningPossibilities
    from evaluation_functions.probabilistic_estimator import ProbabilisticEstimator
    from agents.choose_first_action_agent import ChooseFirstActionAgent
    from agents.hierarchical_agent import HierarchicalAgent
    from agents.random_agent import RandomAgent
    from agents.reinforce_agent import ReinforceAgent
    from agents.search_agents.alpha_beta import AlphaBeta
    from agents.search_agents.mcts import MCTSAgent
    from evaluation_functions.probabilistic_estimator import ProbabilisticEstimator
    from policy_functions.policy_function import PolicyFunction
    from pettingzoo.classic import tictactoe_v3

    parser = argparse.ArgumentParser(description='Evaluate agents and log results')
    parser.add_argument('--log-to-console', default=1, type=int, choices=(0, 1))
    args = parser.parse_args()

    env = ultimate_ttt.env(render_mode=None, depth=1)  # 'human', 'rgb_array', 'ansi', None
    # reinforce_agent: ReinforceAgent = load_agent("../../train/logs/ReinforceAgent/2024-09-04_12-56-44/checkpoint_15000.pickle")
    reinforce_agent: ReinforceAgent = load_agent(
        "../train/logs/ReinforceAgent/2024-09-04_12-56-44/checkpoint_15000.pickle")
    # dqn_agent: DQNAgent = load_agent('../train/logs/DQNAgent/2024-08-31_11-06-41/checkpoint_9.pickle')
    # agents = [
    #     HierarchicalAgent(),
    #     RandomAgent(),
    #     # MCTSAgent(),
    #     # AlphaBeta(2, AEWinningPossibilities()),
    #     AlphaBeta(2, ProbabilisticEstimator()),
    # ]
    mcts_agent = MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent), n_iter_min=20)
    agents = [HierarchicalAgent(), RandomAgent(), ChooseFirstActionAgent(), mcts_agent, AlphaBeta(depth=2, evaluation_function=ProbabilisticEstimator())]
    os.makedirs('logs', exist_ok=True)
    logger_file_name = f"logs/evaluate_agents_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger = get_logger("evaluate_agents", logger_file_name, log_to_console=args.log_to_console)
    agents_evaluator = AgentsEvaluator(agents, env, logger=logger, n_rounds=50)
    agents_evaluator.evaluate_agents()
