from typing import List

import numpy as np
from pettingzoo import AECEnv

from agents.agent import Agent
from agents.choose_first_action_agent import ChooseFirstActionAgent
from agents.random_agent import RandomAgent
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm

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

    print(results)
    env.close()


if __name__ == '__main__':
    players = [RandomAgent(), ChooseFirstActionAgent()]
    env = tictactoe_v3.env(render_mode=None)  # 'human', 'rgb_array', 'ansi', None
    play(env, players)
