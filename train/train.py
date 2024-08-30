import copy
import os
import random
import shutil
from collections import deque
from pathlib import Path
from typing import List, Dict
import matplotlib
import matplotlib.pyplot as plt

from pettingzoo import AECEnv
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm

from agents.trainable_agent import TrainableAgent
from agents.dqn_agent import DQNAgent
from environments import ultimate_ttt

from utils.replay_buffer import ReplayBuffer
from utils.utils import get_action_mask, save_agent

matplotlib.use("TkAgg")


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


def plot_loss(losses: List[float], path_to_save: Path = None):
    abs_losses = np.abs(losses)

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(abs_losses)
    plt.xlabel('Episode')
    plt.ylabel('Absolute Loss')
    plt.ylim(bottom=0)
    plt.title('Absolute Loss per Episode')

    # Save the plot if a path is provided, otherwise show the plot
    if path_to_save:
        plt.savefig(path_to_save)
        plt.close()  # Close the plot to free memory
    else:
        plt.show()


def train(env: AECEnv, agent: TrainableAgent, folder_to_save: Path,
          n_games: int = 10_000,
          agent_pool_size: int = 20,
          add_agent_to_pool_every: int = 500,
          train_every: int = 1,
          replay_buffer_size: int = 10_000,
          discount_factor: float = 0.99,
          seed: int = 42,
          render_every: int = 1000,
          renderable_env: AECEnv = None,
          save_checkpoint_every: int = -1,
          running_loss_alpha: float = 0.9,
          ) -> None:
    """
    train using self-play
    :param env: environment to play in
    :param agent: agent to train vs versions of itself
    :param folder_to_save: file path to save the model to
    :param n_games: number of episodes to train for
    :param agent_pool_size: maximum number of copies of the agent to train against
    :param add_agent_to_pool_every: number of games (episodes) between adding a copy of the existing agent to the pool
    :param train_every: call agent's train_update method every 'train_every' episodes
    :param replay_buffer_size: maximum size for the replay buffer
    :param discount_factor: discount factor for the cumulative reward
    :param seed: seed for reproducibility
    :param render_every: when to render an episode to screen
    :param renderable_env: same as env, but with render_mode='human'. will be displayed render episodes
    :param save_checkpoint_every: when to save a checkpoint of the model, if -1 don't save any checkpoints
    :param running_loss_alpha: alpha for the (exponential) running loss calculation
    """
    assert isinstance(agent, TrainableAgent)
    assert isinstance(folder_to_save, Path), "folder_to_save should be a Path object"
    assert 0 <= discount_factor <= 1
    folder_to_save = folder_to_save / str(agent)
    shutil.rmtree(folder_to_save, ignore_errors=True)
    os.makedirs(folder_to_save, exist_ok=True)
    env.reset(seed=seed)
    agent.train()

    replay_buffer = ReplayBuffer(size=replay_buffer_size)
    losses = []
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
        if curr_game > 0 and curr_game % render_every == 0 and renderable_env is not None:
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
                episode_buffer.append(dict(observation=observation['observation'],
                                           action=action,
                                           action_mask=action_mask,
                                           curr_player_idx=curr_player_idx,
                                           t=len(episode_buffer) // 2))

            env.step(action)
            curr_player_idx = (curr_player_idx + 1) % len(players)

        if curr_game % train_every == 0:
            loss = agent.train_update(replay_buffer)
            if loss is not None:
                running_loss = loss
                if len(losses) > 0:
                    running_loss = running_loss_alpha * loss + (1 - running_loss_alpha) * losses[-1]
                print(f'Loss: {loss:.2f}, Running Loss: {running_loss:.2f}')
                losses.append(running_loss)

        if curr_game > 0 and curr_game % add_agent_to_pool_every == 0:
            previous_agents.append(copy.deepcopy(agent))
            
        if curr_game > 0 and save_checkpoint_every > -1 and curr_game % save_checkpoint_every == 0:
            save_agent(agent, folder_to_save / f'checkpoint_{curr_game // save_checkpoint_every}.pickle')

        if curr_game > 0 and curr_game % render_every == 0 and renderable_env is not None:
            env = original_env

        add_episode_to_replay_buffer(replay_buffer, episode_buffer, discount_factor)
    
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
    import numpy as np
    env = ultimate_ttt.env(render_mode=None, depth=1)
    renderable_env = None  # ultimate_ttt.env(render_mode='human', depth=1)
    env.reset()
    state_shape = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    state_size = np.prod(state_shape)
    hidden_size = 64
    batch_size = 10
    action_size = env.action_space(env.agents[0]).n
    agent = ReinforceAgent(state_size=state_size, action_size=action_size,
                           agent_name=f"ReinforceAgent_depth1_games1e5_hidden{hidden_size}_batch10",
                           hidden_size=hidden_size, batch_size=10)
    folder = Path(__file__).resolve().parent / 'trained_agents'
    train(env, agent, folder, n_games=100_000, render_every=10_000,
          renderable_env=renderable_env, save_checkpoint_every=10_000)
