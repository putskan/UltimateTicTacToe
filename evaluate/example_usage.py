from typing import List

import numpy as np
from pettingzoo import AECEnv

from agents.hierarchical_agent import HierarchicalAgent
from agents.agent import Agent
from agents.random_agent import RandomAgent
from tqdm import tqdm

from environments import ultimate_ttt
from utils.utils import get_action_mask


def play(env: AECEnv, players: List[Agent], n_games: int = 1000, seed: int = 42) -> None:
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
        env.reset()
        for player in players:
            player.reset()
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

        is_draw = cumulative_rewards[0] == cumulative_rewards[1]
        if is_draw:
            results['draws'] += 1
        else:
            winner = np.argmax(cumulative_rewards).item()
            results[f'{players[winner]} wins'] += 1

    print(results)
    env.close()


if __name__ == '__main__':
    players = [HierarchicalAgent(), RandomAgent()]
    env = ultimate_ttt.env(render_mode='human', depth=2)
    play(env, players, n_games=1)
