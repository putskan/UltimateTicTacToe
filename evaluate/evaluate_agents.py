import os
from typing import List, Tuple
from itertools import permutations
from collections import Counter
from logging import Logger
import datetime
import argparse

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


def play_single_game(env: AECEnv, players: Tuple[Agent, Agent]) -> Tuple[float, float]:
    """
    play n_games between the players and print the results
    :param env: env to play in
    :param players: 2 players to play the game
    :return: results of the games - score for each player: win - 1, draw - 0.5, lose - 0
    """
    cumulative_rewards = [0] * len(players)
    curr_player_idx = 0
    env.reset()
    for curr_agent_str in env.agent_iter():
        curr_player = players[curr_player_idx]
        observation, reward, termination, truncation, info = env.last()
        cumulative_rewards[curr_player_idx] += reward
        action_mask = get_action_mask(observation, info)
        done = termination or truncation
        if done:
            action = None
        else:
            action = curr_player.play(env, observation, curr_player_idx, curr_agent_str, action_mask, info)
        env.step(action)
        curr_player_idx = (curr_player_idx + 1) % len(players)

    # scores: win - 1, draw - 0.5, lose - 0
    score1 = (cumulative_rewards[0] + 1) / 2
    score2 = (cumulative_rewards[1] + 1) / 2
    env.close()
    return score1, score2


def log_final_results(logger: Logger, title: str, final_results: List[Tuple[str, float]], use_percentage: bool = False) -> None:
    """
    Logs the final results of the evaluation
    :param logger: logger to log results
    :param title: title of the results
    :param final_results: list of tuples containing the agent name and the winning percentage, sorted by winning percentage
    """
    logger.info("************************************")
    logger.info(f"Final {title} Results:")
    for agent_name, winning_percentage in final_results:
        msg = f"{agent_name}: {winning_percentage:.2f}"
        if use_percentage:
            msg = f"{agent_name}: {winning_percentage:.2%}"
        logger.info(msg)
    logger.info("************************************")
    logger.info("")


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


def evaluate_agents(env: AECEnv, agents: List[Agent], logger: Logger, n_games: int = 1000) -> None:
    """
    Evaluates each agent by the percentage of winning games (or draws) against the other agents
    :param env: env to play in
    :param agents: list of agents to evaluate
    :param logger: logger to log results
    :param n_games: number of games to play
    """
    player_win = Counter()
    player_elo_rating = {agent: INITIAL_ELO_RATE for agent in agents}

    for _ in tqdm(range(n_games)):
        for agent1, agent2 in permutations(agents, 2):
            score1, score2 = play_single_game(env, (agent1, agent2))

            # update win count of each player
            player_win[agent1] += score1
            player_win[agent2] += score2

            # update elo ranting of each player
            rate1, rate2 = player_elo_rating[agent1], player_elo_rating[agent2]
            new_rate1, new_rate2 = calc_new_elo_rating(rate1, rate2, score1, score2)
            player_elo_rating[agent1] = new_rate1
            player_elo_rating[agent2] = new_rate2

    # divide by the number of games each player played to get the average winning percentage
    for player in player_win:
        player_win[player] /= 2 * (len(agents) - 1) * n_games
    assert sum(player_win.values()) == len(agents) / 2

    # log results
    sorted_elo_ratings = sorted(player_elo_rating.items(), key=lambda item: item[1], reverse=True)
    log_final_results(logger, "Winning Percentage", player_win.most_common(), use_percentage=True)
    log_final_results(logger, "Elo Rating", sorted_elo_ratings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate agents and log results')
    parser.add_argument('--log-to-console', default=1, type=int, choices=(0, 1))
    args = parser.parse_args()

    # env = tictactoe_v3.env(render_mode=None)  # 'human', 'rgb_array', 'ansi', None
    env = ultimate_ttt.env(render_mode=None, depth=2)  # 'human', 'rgb_array', 'ansi', None
    players = [RandomAgent(), ChooseFirstActionAgent(),
               # UnbeatableClassicTTTAgent(agent_name='unbeatable1'), UnbeatableClassicTTTAgent(agent_name='unbeatable2'),
               HierarchicalAgent(),
               ]

    os.makedirs('logs', exist_ok=True)
    logger_file_name = f"logs/evaluate_agents_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger = get_logger("evaluate_agents", logger_file_name, log_to_console=args.log_to_console)
    evaluate_agents(env, players, logger)
