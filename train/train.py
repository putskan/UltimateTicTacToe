import copy
import random
from collections import deque
from typing import List, Dict

from pettingzoo import AECEnv
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm

from agents.trainable_agent import TrainableAgent
from agents.dqn_agent import DQNAgent
from environments import ultimate_ttt

from utils.replay_buffer import ReplayBuffer
from utils.utils import get_action_mask


def add_episode_to_replay_buffer(replay_buffer: ReplayBuffer,
                                 episode_buffer: List[Dict],
                                 discount_factor: float) -> None:
    cumulative_reward = [0., 0.]
    for i in range(len(episode_buffer) - 1, -1, -1):
        curr_record = episode_buffer[i]
        curr_player_idx = curr_record['curr_player_idx']
        curr_reward = curr_record['reward']
        cumulative_reward[curr_player_idx] = curr_reward + discount_factor * cumulative_reward[curr_player_idx]
        curr_record['cumulative_reward'] = cumulative_reward[curr_player_idx]

    for curr_record in episode_buffer:
        replay_buffer.push(**curr_record)


def train(env: AECEnv, agent: TrainableAgent, n_games: int = 10_000,
          agent_pool_size: int = 20,
          add_agent_to_pool_every: int = 500,
          train_every: int = 1,
          replay_buffer_size: int = 10_000,
          discount_factor: float = 0.99,
          seed: int = 42,
          render_every: int = 1000,
          renderable_env: AECEnv = None,
          ) -> None:
    """
    train using self-play
    :param env: environment to play in
    :param agent: agent to train vs versions of itself
    :param n_games: number of episodes to train for
    :param agent_pool_size: maximum number of copies of the agent to train against
    :param add_agent_to_pool_every: number of games (episodes) between adding a copy of the existing agent to the pool
    :param train_every: call agent's train_update method every 'train_every' episodes
    :param replay_buffer_size: maximum size for the replay buffer
    :param discount_factor: discount factor for the cumulative reward
    :param seed: seed for reproducibility
    :param render_every: when to render an episode to screen
    :param renderable_env: same as env, but with render_mode='human'. will be displayed render episodes
    """
    assert isinstance(agent, TrainableAgent)
    assert 0 <= discount_factor <= 1
    env.reset(seed=seed)
    agent.train()

    replay_buffer = ReplayBuffer(size=replay_buffer_size)
    previous_agents = deque(maxlen=agent_pool_size)
    previous_agents.append(copy.deepcopy(agent))
    for curr_game in tqdm(range(n_games)):
        episode_buffer = []
        main_agent_idx = random.randint(0, 1)
        if main_agent_idx == 0:
            players = [agent, random.choice(previous_agents)]
        else:
            players = [random.choice(previous_agents), agent]

        curr_player_idx = 0
        if curr_game > 0 and curr_game % render_every == 0:
            original_env = env
            env = renderable_env

        env.reset()
        for curr_agent_str in env.agent_iter():
            curr_player = players[curr_player_idx]
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            action_mask = get_action_mask(observation, info)

            if len(episode_buffer) >= 2:  # skip first ones
                idx = -2
                if episode_buffer[-2]['curr_player_idx'] != curr_player_idx:
                    assert done
                    idx = -1

                episode_buffer[idx].update({
                    'next_observation': observation,
                    'reward': reward,
                    'done': done,
                    'next_action_mask': action_mask,
                })

            if done:
                action = None
            else:
                action = curr_player.play(env, observation, curr_player_idx, curr_agent_str, action_mask, info)
                episode_buffer.append(dict(observation=observation,
                                           action=action,
                                           action_mask=action_mask,
                                           curr_player_idx=curr_player_idx))

            env.step(action)
            curr_player_idx = (curr_player_idx + 1) % len(players)

        if curr_game % train_every == 0:
            agent.train_update(replay_buffer)

        if curr_game > 0 and curr_game % add_agent_to_pool_every == 0:
            previous_agents.append(copy.deepcopy(agent))

        if curr_game > 0 and curr_game % render_every == 0 and renderable_env is not None:
            env = original_env

        add_episode_to_replay_buffer(replay_buffer, episode_buffer, discount_factor)

    env.close()


if __name__ == '__main__':
    # env = ultimate_ttt.env(render_mode='human', depth=2, render_fps=10)
    env = tictactoe_v3.env(render_mode=None)
    renderable_env = tictactoe_v3.env(render_mode='human')
    env.reset()
    state_size = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    action_size = env.action_space(env.agents[0]).n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    train(env, agent, n_games=100_000, render_every=20_000, renderable_env=renderable_env)
