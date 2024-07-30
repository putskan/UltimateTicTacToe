from typing import List, Tuple, Dict
from itertools import permutations
from collections import Counter
from logging import Logger
import datetime
import argparse

import numpy as np
from pettingzoo import AECEnv

from agents.agent import Agent
from agents.choose_first_action_agent import ChooseFirstActionAgent
from agents.random_agent import RandomAgent
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm

from utils.utils import get_action_mask
from utils.logger import get_logger


def play(env: AECEnv, players: Tuple[Agent, Agent], n_games: int = 1000, seed: int = 42) -> Dict[str, int]:
    """
    play n_games between the players and print the results
    :param env: env to play in
    :param players: list of agents
    :param n_games: number of games to play
    :param seed: seed for reproducibility
    """
    env.reset(seed=seed)
    results = {f'{player} wins': 0 for player in players}
    results['draws'] = 0

    for _ in tqdm(range(n_games)):
        cumulative_rewards = [0] * len(players)
        curr_player_idx = 0
        obs = env.reset()
        for curr_agent_str in env.agent_iter():
            curr_player = players[curr_player_idx]
            observation, reward, termination, truncation, info = env.last()
            cumulative_rewards[curr_player_idx] += reward
            action_mask = get_action_mask(observation, info)
            done = termination or truncation
            if done:
                action = None
            else:
                action = curr_player.play(env, obs, curr_player_idx, curr_agent_str, action_mask)
            env.step(action)
            curr_player_idx = (curr_player_idx + 1) % len(players)

        is_draw = np.all(np.array(cumulative_rewards) == cumulative_rewards[0])
        if is_draw:
            results['draws'] += 1
        else:
            winner = np.argmax(cumulative_rewards).item()
            results[f'{players[winner]} wins'] += 1

    env.close()
    return results


def log_results(logger: Logger, player1_name: str, player2_name: str, results: Dict[str, int]) -> None:
    """
    Logs the results of a single play
    :param player1_name: name of first player
    :param player2_name: name of second player
    :param results: dictionary containing results of play
    """
    logger.info(f"Results for play of {player1_name} vs. {player2_name}:")
    logger.info(f">>>>>> {results}")


def log_final_results(logger: Logger, final_results: List[Tuple[str, float]]) -> None:
    """
    Logs the final results of the evaluation
    :param logger: logger to log results
    :param final_results: list of tuples containing the agent name and the winning percentage, sorted by winning percentage
    """
    logger.info("")
    logger.info("************************************")
    logger.info("Final Results:")
    for agent_name, winning_percentage in final_results:
        logger.info(f"{agent_name}: {winning_percentage:.3%}")
    logger.info("************************************")


def evaluate_agents_by_winning_percentage(env: AECEnv, agents: List[Agent], logger: Logger,
                                          n_games: int = 1000, include_draws: bool = True) -> List[Tuple[str, float]]:
    """
    Evaluates each agent by the percentage of winning games (or draws) against the other agents
    :param env: env to play in
    :param agents: list of agents to evaluate
    :param logger: logger to log results
    :param n_games: number of games to play
    :param include_draws: whether to include draws in the evaluation
    :return: a list of tuples containing the agent name and the winning percentage, sorted by winning percentage
    """
    player_rank = Counter()

    for agent1, agent2 in permutations(agents, 2):
        results = play(env, (agent1, agent2), n_games)
        log_results(logger, str(agent1), str(agent2), results)

        # update ranking of each player
        player_rank[agent1] += results[f'{agent1} wins']
        player_rank[agent2] += results[f'{agent2} wins']
        if include_draws:
            player_rank[agent1] += 0.5 * results['draws']
            player_rank[agent2] += 0.5 * results['draws']

    # divide by the number of games each player played to get the average winning percentage
    for player in player_rank:
        player_rank[player] /= 2 * (len(agents) - 1) * n_games
    assert not include_draws or sum(player_rank.values()) == len(agents) / 2
    return player_rank.most_common()


def evaluate_agents_by_elo_rank(env: AECEnv, agents: List[Agent]):
    # TODO: implement
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate agents and log results')
    parser.add_argument('--log-to-console', default=1, type=int, choices=(0, 1))
    args = parser.parse_args()
    logger_file_name = f"logs/evaluate_agents_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger = get_logger('evaluate_agents', logger_file_name, log_to_console=args.log_to_console)

    players = [RandomAgent("random1"), ChooseFirstActionAgent("choose1"),
               RandomAgent("random2"), ChooseFirstActionAgent("choose2")]
    # TODO: switch to U-TTT environment
    env = tictactoe_v3.env(render_mode=None)  # 'human', 'rgb_array', 'ansi', None
    # TODO: maybe add option to evaluate by sampling opponents randomly (to speed up the process)
    # TODO: evaluate by winning percentage vs elo ranking
    final_rankings = evaluate_agents_by_winning_percentage(env, players, logger)
    log_final_results(logger, final_rankings)
