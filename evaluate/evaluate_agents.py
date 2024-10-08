import re
from pathlib import Path
from typing import List, Tuple, Dict, Union
from itertools import permutations
from logging import Logger
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from pettingzoo import AECEnv


from utils.utils import get_action_mask
from utils.logger import get_logger
from agents.agent import Agent
from agents.timed_agent_wrapper import TimedAgentWrapper

matplotlib.use("TkAgg")


K = 32
s = 400
INITIAL_ELO_RATE = 1200


class AgentsEvaluator:
    """
    Class to evaluate agents' performance by playing multiple rounds of games
    """
    def __init__(self, agents: List[Agent], env: AECEnv, logger: Logger = None, n_rounds: int = 1000,
                 from_file: Union[Path, str] = None):
        """
        :param agents: list of agents to evaluate
        :param env: env to play in
        :param logger: logger to log results. if None, default logger is created and logs only to console
        :param n_rounds: number of rounds to play.
                        for n_rounds=n and len(agents)=k, we play n * (k * (k-1)) games
        """
        agent_strs = [str(agent) for agent in agents]
        assert len(agent_strs) == len(np.unique(agent_strs)), 'Agent names must unique'
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
        self.wins_as_player_1 = np.zeros((n_agents, n_agents), dtype=int)
        self.draws_as_player_1 = np.zeros((n_agents, n_agents), dtype=int)
        self.players_elo_rating = {agent: INITIAL_ELO_RATE for agent in self.agents}
        self.player_total_games = 2 * (len(self.agents) - 1) * self.n_rounds
        self.from_file = from_file

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

    def _agent_str_to_agent(self, agent_str: str) -> Tuple[int, Union[Agent, TimedAgentWrapper]]:
        for idx, agent in enumerate(self.agents):
            if agent_str == str(agent):
                return idx, agent
        raise ValueError(f'Agent {agent_str} not found')

    def evaluate_agents(self) -> None:
        """
        Run games between the agents and assess their performance
        """
        n_agents = len(self.agents)

        if self.from_file:
            pattern = re.compile(r'DEBUG - (.+) vs (.+): ([\d.]{3})-([\d.]{3})')
            n_games = 0
            with open(self.from_file, 'r') as f:
                for line in f:
                    if ' vs ' not in line:
                        continue
                    n_games += 1
                    match = pattern.search(line)
                    agent1_str = match.group(1)
                    agent2_str = match.group(2)
                    score1 = float(match.group(3))
                    score2 = float(match.group(4))
                    agent1_idx, agent1 = self._agent_str_to_agent(agent1_str)
                    agent2_idx, agent2 = self._agent_str_to_agent(agent2_str)
                    self.update_stats(agent1_idx, agent2_idx, agent1, agent2, score1, score2)
            actual_n_rounds = int(n_games / (len(self.agents) * (len(self.agents) - 1)))
            assert actual_n_rounds == self.n_rounds, \
                f'file is inconsistent with config. change n_rounds to {actual_n_rounds}'

        else:
            all_agent_pairs = list(permutations(range(n_agents), 2))
            for _ in tqdm(range(self.n_rounds)):
                for agent1_idx, agent2_idx in all_agent_pairs:
                    agent1, agent2 = self.agents[agent1_idx], self.agents[agent2_idx]
                    score1, score2 = self.play_single_game((agent1, agent2))
                    self.logger.debug(f'{agent1} vs {agent2}: {score1}-{score2}')
                    self.update_stats(agent1_idx, agent2_idx, agent1, agent2, score1, score2)

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
        self.plot_results(player_avg_time)

    @property
    def log_folder(self):
        """
        :return: the folder of the logger, if exists
        """
        for handler in self.logger.handlers:
            if hasattr(handler, 'baseFilename'):
                return Path(handler.baseFilename).parent

    def _set_plot_metadata_and_save(self, title: str, save_plot: bool, suptitle: str = None, pad_title: int = None):
        if pad_title is not None:
            plt.title(title, pad=pad_title)
        else:
            plt.title(title)
        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=14)

        if self.log_folder and save_plot:
            dst_path = self.log_folder / f'{title}.png'
            plt.savefig(dst_path, bbox_inches='tight')
            self.logger.info(f'Saved {dst_path} successfully')

    @staticmethod
    def _plot_heatmap(df: pd.DataFrame, cmap: str, x_rotation: int = 90, fmt: str = "d"):
        heatmap = sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, cbar=False)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=x_rotation)

    def plot_results(self, player_avg_time: Dict[str, int], save_plot: bool = True) -> None:
        """
        Plot the results from the agent evaluation process
        Optionally also saves the plots to the logger's folder.
        :param player_avg_time: the avg time for each player to make a move
        :param save_plot: whether to save the plot or not
        """
        draws = (2 * self.n_rounds) - (self.wins + self.wins.T)
        draws[np.eye(len(draws), dtype=bool)] = 0
        # Creating a DataFrame for each matrix for better visualization
        wins_df = pd.DataFrame(self.wins, index=self.agents, columns=self.agents)
        losses_df = pd.DataFrame(self.wins.T, index=self.agents, columns=self.agents)
        draws_df = pd.DataFrame(draws, index=self.agents, columns=self.agents)

        # Visualizing the matrices using heatmaps

        plt.figure(1, figsize=(10, 8))
        self._plot_heatmap(wins_df, cmap="Blues")
        self._set_plot_metadata_and_save('Wins', save_plot=save_plot)

        plt.figure(2, figsize=(10, 8))
        self._plot_heatmap(losses_df, cmap="Reds")
        self._set_plot_metadata_and_save('Losses', save_plot=save_plot)

        plt.figure(3, figsize=(10, 8))
        self._plot_heatmap(draws_df, cmap="Greens")
        self._set_plot_metadata_and_save('Draws', save_plot=save_plot)

        # bar plot for total scores
        plt.figure(4, figsize=(16, 6))
        total_scores = wins_df.sum(axis=1) + draws_df.sum(axis=1) * 0.5  # Wins + 0.5 * Draws
        colors = sns.color_palette("crest", len(self.wins))

        total_scores.sort_values().plot(kind='barh', color=colors)
        plt.xlabel('Score')
        plt.ylabel('Agent')
        self._set_plot_metadata_and_save('Total Scores by Agent', save_plot=save_plot)

        # Elo rating
        plt.figure(5, figsize=(16, 6))
        elo_ratings = pd.Series(self.players_elo_rating).sort_values()
        elo_ratings.plot(kind='barh', color=colors)
        plt.xlabel('Elo Rating')
        plt.ylabel('Agent')
        suptitle = f'Game Results Between Agents ({2 * self.n_rounds} Games per Pair)'
        self._set_plot_metadata_and_save('Elo Ratings by Agent', save_plot=save_plot, suptitle=suptitle)

        self.plot_average_times(player_avg_time, figure_num=6)

        wins_as_player_1_df = pd.DataFrame(self.wins_as_player_1, index=self.agents, columns=self.agents)
        draws_as_player_1_df = pd.DataFrame(self.draws_as_player_1, index=self.agents, columns=self.agents)
        plt.figure(7, figsize=(10, 8))
        self._plot_heatmap(wins_as_player_1_df, cmap="Purples")
        self._set_plot_metadata_and_save('Wins as X', save_plot=save_plot)

        plt.figure(8, figsize=(10, 8))
        wins_as_player_1_pct_df = wins_as_player_1_df / wins_df
        np.fill_diagonal(wins_as_player_1_pct_df.values, 0)
        self._plot_heatmap(wins_as_player_1_pct_df, fmt=".2f", cmap="Purples")
        self._set_plot_metadata_and_save('Wins as X (as % of all wins)', save_plot=save_plot)

        plt.figure(9, figsize=(10, 8))
        self._plot_heatmap(draws_as_player_1_df, cmap="Oranges")
        self._set_plot_metadata_and_save('Draws as X', save_plot=save_plot)

        plt.figure(10, figsize=(10, 8))
        draws_as_player_1_pct_df = draws_as_player_1_df / draws_df
        np.fill_diagonal(draws_as_player_1_pct_df.values, 0)
        self._plot_heatmap(draws_as_player_1_pct_df, fmt=".2f", cmap="Oranges")
        self._set_plot_metadata_and_save('Draws as X (as % of all draws)', save_plot=save_plot)

        games_x_won = wins_as_player_1_df.sum().sum()
        games_o_won = wins_df.sum().sum() - games_x_won
        games_drawn = draws_df.sum().sum() // 2
        total_games = games_drawn + games_x_won + games_o_won
        assert games_drawn + games_x_won + games_o_won == self.n_rounds * len(self.agents) * (len(self.agents) - 1)
        assert draws_df.sum().sum() % 2 == 0

        labels = ['X Won', 'O Won', 'Draws']
        sizes = [games_x_won, games_o_won, games_drawn]
        colors = sns.color_palette('pastel')[0:3]  # Choose a color palette from Seaborn
        # Create the pie chart
        plt.figure(11, figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors,
                autopct=lambda x: str(int(round(x / 100 * total_games))), startangle=140)
        self._set_plot_metadata_and_save(f'Win by Piece (Total: {total_games} Games)',
                                         save_plot=save_plot, pad_title=20)

        # ensures that pie is drawn as a circle.
        plt.axis('equal')
        plt.show()

    def plot_average_times(self, agents_times: Dict[str, float], figure_num: int, save_plot: bool = True) -> None:
        """
        Plots the results as 3 heatmaps of wins, loses and draws.
        Optionally also saves the plots to the logger's folder.
        :param agents_times: a dictionary of agent names and their average time per move
        :param save_plot: whether to save the plot or not
        """
        agents_times = pd.Series(agents_times).sort_values()
        palette = sns.color_palette("crest")

        plt.figure(figure_num, figsize=(13, 10))
        bars = plt.bar(agents_times.index, agents_times.values, color=palette)
        plt.ylabel('Average Time (seconds)')
        plt.gca().xaxis.set_ticks(agents_times.index)
        plt.gca().set_xticklabels(agents_times.index, rotation=20, ha='right', fontsize=8)

        # Add the value labels on top of each bar
        for bar, value in zip(bars, agents_times.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X position: center of the bar
                bar.get_height(),
                f'{value:.2f}',
                ha='center',
                va='bottom'
            )

        self._set_plot_metadata_and_save('Average Time Per Agent', save_plot=save_plot)

    def update_stats(self, agent1_idx: int, agent2_idx: int, agent1: Union[TimedAgentWrapper, Agent],
                     agent2: Union[TimedAgentWrapper, Agent], score1: float, score2: float) -> None:
        # update win count of each player
        self.wins[agent1_idx, agent2_idx] += score1 == 1
        self.wins[agent2_idx, agent1_idx] += score2 == 1

        self.wins_as_player_1[agent1_idx, agent2_idx] += score1 == 1
        self.draws_as_player_1[agent1_idx, agent2_idx] += score1 == 0.5

        # update elo ranting of each player
        rate1, rate2 = self.players_elo_rating[agent1], self.players_elo_rating[agent2]
        new_rate1, new_rate2 = self.calc_new_elo_rating(rate1, rate2, score1, score2)
        self.players_elo_rating[agent1] = new_rate1
        self.players_elo_rating[agent2] = new_rate2
