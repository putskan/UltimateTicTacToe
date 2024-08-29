import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from models.reinforce import ReinforcePolicy

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

policy = ReinforcePolicy(4, 2, 128)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    policy_vals = policy(state)
    probs = F.softmax(policy_vals, dim=-1)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


def finish_episode(rewards, log_probs):
    R = 0
    returns = deque()
    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    log_probs = torch.cat(log_probs)
    policy_loss = -log_probs * returns
    policy_loss = policy_loss.sum()
    optimizer.zero_grad()
    policy_loss.backward()

    optimizer.step()


def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        rewards = []
        log_probs = []
        for t in range(1, 10000):  # Don't infinite loop while learning
            action, log_prob = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            rewards.append(reward)
            log_probs.append(log_prob)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(rewards, log_probs)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
