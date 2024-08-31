import argparse
from collections import deque

import gymnasium as gym
import numpy as np
from itertools import count
import torch

from agents.dqn_agent import DQNAgent
# from agents.reinforce_agent import ReinforceAgent
# from models.reinforce import ReinforcePolicy
from train.train import add_episode_to_replay_buffer
from utils.replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

policy = DQNAgent(4, 2, 128, learning_rate=1e-3)
eps = np.finfo(np.float32).eps.item()


def main(replay_buffer_size: int = 10_000, discount_factor: float = 0.99):
    running_reward = 10
    replay_buffer = ReplayBuffer(size=replay_buffer_size)

    for i_episode in count(1):
        state, _ = env.reset()
        episode_buffer = []
        ep_reward = 0
        actions = deque(maxlen=20)
        for t in range(1, 10000):  # Don't infinite loop while learning
            policy.train_update(replay_buffer)

            action_mask = np.ones(2).astype(bool)
            # TODO: add next observation and next action mask to episode_info correctly
            episode_info = dict(observation=state, action_mask=action_mask, curr_player_idx=0, t=t,
                                next_action_mask=action_mask)
            action = policy.play(env, dict(observation=state), 0, '', action_mask, None)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            ep_reward += reward
            episode_info.update({
                'action': action,
                'reward': reward,
                'done': done,
                'next_observation': state
            })
            episode_buffer.append(episode_info)
            actions.append(action)
            if done:
                break

        add_episode_to_replay_buffer(replay_buffer, episode_buffer, discount_factor)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLast actions mean: {:.2f}'.format(
                  i_episode, ep_reward, running_reward, np.mean(actions)))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()