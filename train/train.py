import copy
import datetime
import logging
import os
import random
from collections import deque
from pathlib import Path
from typing import List, Dict, Union
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
from pettingzoo import AECEnv
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm

import agents
from agents.agent import Agent
from agents.choose_first_action_agent import ChooseFirstActionAgent
from agents.hierarchical_agent import HierarchicalAgent
from agents.random_agent import RandomAgent
from agents.trainable_agent import TrainableAgent
from agents.dqn_agent import DQNAgent
from models.dqn import DQN, PrevDQN, DuelingDQN
from utils.logger import get_logger

from utils.replay_buffer import ReplayBuffer
from utils.utils import get_action_mask, save_agent

matplotlib.use("TkAgg")


def evaluate(env: AECEnv, agent: TrainableAgent, n_games: int = 100, seed: int = 42,
             opponent_agent: Agent = ChooseFirstActionAgent()) -> float:
    """
    evaluate the agent
    :param env: environment to play in
    :param agent: agent to evaluate
    :param n_games: number of episodes to evaluate for
    :param seed: seed for reproducibility
    :param opponent_agent: opponent to evaluate against
    :return: average reward over the n_games episodes
    """
    agent.eval()
    with torch.no_grad():
        env.reset(seed=seed)
        agent.eval()
        total_reward = 0
        for _ in range(n_games):
            env.reset()

            main_agent_idx = random.randint(0, 1)
            if main_agent_idx == 0:
                players = [agent, opponent_agent]
            else:
                players = [opponent_agent, agent]

            curr_player_idx = 0
            cumulative_rewards = [0, 0]

            for curr_agent_str in env.agent_iter():
                curr_player = players[curr_player_idx]

                observation, reward, termination, truncation, info = env.last()
                done = termination or truncation
                if done:
                    action = None
                else:
                    action_mask = get_action_mask(observation, info)
                    with torch.no_grad():
                        action = curr_player.play(env, observation, curr_player_idx, curr_agent_str, action_mask, info)

                env.step(action)
                cumulative_rewards[curr_player_idx] += reward
                curr_player_idx = (curr_player_idx + 1) % len(players)

            is_draw = cumulative_rewards[0] == cumulative_rewards[1]
            if is_draw:
                total_reward += 0.5

            else:
                winner = np.argmax(cumulative_rewards).item()
                if winner == main_agent_idx:
                    total_reward += 1
        return total_reward / n_games


def add_episode_to_replay_buffer(replay_buffer: ReplayBuffer,
                                 episode_buffer: List[Dict],
                                 discount_factor: float,
                                 main_agent_idx: int,
                                 main_agent_only: bool) -> None:
    cumulative_reward = [0., 0.]
    for i in range(len(episode_buffer) - 1, -1, -1):
        curr_record = episode_buffer[i]
        curr_player_idx = curr_record['curr_player_idx']
        curr_reward = curr_record['reward']
        cumulative_reward[curr_player_idx] = curr_reward + discount_factor * cumulative_reward[curr_player_idx]
        curr_record['cumulative_reward'] = cumulative_reward[curr_player_idx]

    for curr_record in episode_buffer:
        if (not main_agent_only) or curr_record['curr_player_idx'] == main_agent_idx:
            replay_buffer.push(**curr_record)


def plot_loss(losses: List[float], path_to_save: Path = None, abs_loss: bool = True):
    """
    :param losses:
    :param path_to_save:
    :return:
    """
    if abs_loss:
        losses = np.abs(losses)

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Episode')
    abs_str = {"Absolute" if abs_loss else ""}
    plt.ylabel(f'{abs_str}Loss')
    plt.ylim(bottom=0)
    plt.title(f'{abs_str}Loss per Episode')

    if path_to_save:
        plt.savefig(path_to_save)

    plt.show()
    plt.close()


def train(env: AECEnv, agent: TrainableAgent,
          n_games: int = 10_000,
          agent_pool_size: int = 10,
          add_agent_to_pool_every: int = 500,
          train_every: int = 1,
          replay_buffer_size: int = 10_000,
          discount_factor: float = 0.99,
          seed: int = 42,
          render_every: int = 1000,
          renderable_env: AECEnv = None,
          evaluate_every: int = 100,
          only_main_agent_to_replay: bool = False,
          save_checkpoint_every: int = -1,
          running_loss_alpha: float = 0.9,
          folder_to_save: Union[Path, str] = None,
          logger: logging.Logger = None) -> None:
    """
    train using self-play
    :param env: environment to play in
    :param agent: agent to train vs versions of itself
    :param folder_to_save: file path to save the model to
    :param n_games: number of episodes to train for
    :param agent_pool_size: maximum number of copies of the agent to train against
    :param add_agent_to_pool_every: number of games (episodes) between adding a copy of the existing agent to the pool
    :param train_every: call agent's train_update method every 'train_every' steps
    :param replay_buffer_size: maximum size for the replay buffer
    :param discount_factor: discount factor for the cumulative reward
    :param seed: seed for reproducibility
    :param render_every: when to render an episode to screen
    :param renderable_env: same as env, but with render_mode='human'. will be displayed render episodes
    :param save_checkpoint_every: when to save a checkpoint of the model, if -1 don't save any checkpoints
    :param running_loss_alpha: alpha for the (exponential) running loss calculation
    :param logger: optional logger object
    """
    assert isinstance(agent, TrainableAgent)
    assert 0 <= discount_factor <= 1
    date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if folder_to_save is not None:
        folder_to_save = Path(folder_to_save) / str(agent) / date_str
        os.makedirs(folder_to_save, exist_ok=True)

    if logger is None:
        logger = get_logger(__name__, log_dir_name=folder_to_save, log_to_console=folder_to_save is None)

    agent.train()
    env.reset(seed=seed)
    replay_buffer = ReplayBuffer(size=replay_buffer_size)
    losses = []
    previous_agents = deque(maxlen=agent_pool_size)
    previous_agents.append(copy.deepcopy(agent))
    pbar = tqdm(range(n_games))
    last_eval = None
    log_dict = {}
    for curr_game in pbar:
        episode_buffer = []
        main_agent_idx = random.randint(0, 1)
        if main_agent_idx == 0:
            players = [agent, random.choice(previous_agents)]
        else:
            players = [random.choice(previous_agents), agent]

        for player in players:
            player.train()

        curr_player_idx = 0
        if curr_game > 0 and curr_game % render_every == 0 and renderable_env is not None:
            original_env = env
            env = renderable_env

        env.reset()
        for curr_agent_str in env.agent_iter():
            if curr_game % train_every == 0:
                train_data = agent.train_update(replay_buffer)
                log_dict.update({'last_eval': last_eval, **(train_data or {})})
                if isinstance(train_data, dict):
                    running_loss = train_data['loss']
                    if len(losses) > 0:
                        running_loss = running_loss_alpha * train_data['loss'] + (1 - running_loss_alpha) * losses[-1]
                    losses.append(running_loss)

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
                    'next_observation': observation['observation'],
                    'reward': reward,
                    'done': done,
                    'next_action_mask': action_mask,
                })

            if done:
                action = None
            else:
                with torch.no_grad():
                    action = curr_player.play(env, observation, curr_player_idx, curr_agent_str, action_mask, info)

                episode_buffer.append(dict(observation=observation['observation'],
                                           action=action,
                                           action_mask=action_mask,
                                           curr_player_idx=curr_player_idx,
                                           t=len(episode_buffer) // 2))

            env.step(action)
            curr_player_idx = (curr_player_idx + 1) % len(players)

        if curr_game % train_every == 0:
            agent.train_update(replay_buffer)

        if curr_game > 0 and curr_game % add_agent_to_pool_every == 0:
            previous_agents.append(copy.deepcopy(agent))

        if curr_game > 0 and save_checkpoint_every > -1 and curr_game % save_checkpoint_every == 0:
            save_agent(agent, folder_to_save / f'checkpoint_{curr_game}.pickle')

        if curr_game > 0 and curr_game % render_every == 0 and renderable_env is not None:
            env = original_env

        if curr_game > 0 and curr_game % evaluate_every == 0:
            last_eval = evaluate(env, agent, n_games=10, seed=seed)

        add_episode_to_replay_buffer(replay_buffer, episode_buffer, discount_factor,
                                     main_agent_idx, only_main_agent_to_replay)
        logger.info(log_dict)
        pbar.set_postfix(log_dict)

    save_agent(agent, folder_to_save / 'final.pickle')
    if losses:
        plot_loss(losses, folder_to_save / 'loss.png')
    env.close()


if __name__ == '__main__':
    # TODO: fix state (observation) unpacking in DQN agent
    # # DQN training
    # # env = ultimate_ttt.env(render_mode='human', depth=2, render_fps=10)
    # env = tictactoe_v3.env(render_mode=None)
    # renderable_env = tictactoe_v3.env(render_mode='human')
    # env.reset()
    # state_size = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    # action_size = env.action_space(env.agents[0]).n
    # agent = DQNAgent(state_size=state_size, action_size=action_size)
    # train(env, agent, n_games=100_000, render_every=20_000, renderable_env=renderable_env)

    # REINFORCE training
    from agents.reinforce_agent import ReinforceAgent
    from environments import ultimate_ttt
    import numpy as np
    env = ultimate_ttt.env(render_mode=None, depth=1)
    renderable_env = None  # ultimate_ttt.env(render_mode='human', depth=1)
    env.reset()
    state_shape = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    state_size = np.prod(state_shape)
    hidden_size = 64
    batch_size = 10
    action_size = env.action_space(env.agents[0]).n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    train(env, agent, n_games=100_000, render_every=20_000, renderable_env=renderable_env)
