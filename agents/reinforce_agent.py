from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch import Tensor

from agents.trainable_agent import TrainableAgent
from models.reinforce import ReinforcePolicy
from utils.replay_buffer import ReplayBuffer


class ReinforceAgent(TrainableAgent):
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=1e-4, gamma=0.99,
                 batch_size=64, replay_buffer_size=10000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = ReinforcePolicy(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.modules.append(self.policy_net)

    def play(self, env, obs, curr_agent_idx, curr_agent_str, action_mask, info: Dict[str, Any]):
        state = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)
        action_mask = torch.BoolTensor(action_mask).to(self.device).unsqueeze(0)
        actions, _ = self._select_actions(state, action_mask)
        return actions.item()

    def _select_actions(self, states: Tensor, action_masks: Tensor) -> Tuple[Tensor, Tensor]:
        policy_vals = self.policy_net(states)
        policy_vals[action_masks.bitwise_not()] = -float('inf')
        probs = nn.functional.softmax(policy_vals, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        return actions, m.log_prob(actions)

    def train_update(self, replay_buffer: ReplayBuffer):
        if len(replay_buffer) < self.batch_size:
            print("Replay buffer not large enough to train")
            return

        batch = replay_buffer.sample(self.batch_size)

        (states, _, rewards, _, dones, action_masks,
         curr_player_idxs, _, cumulative_rewards, t) = zip(*batch)
        # states = torch.stack([torch.FloatTensor(state['observation']) for state in states]).to(self.device)
        states = torch.FloatTensor(states).to(self.device)
        action_masks = torch.BoolTensor(action_masks).to(self.device)
        cumulative_rewards = torch.FloatTensor(cumulative_rewards).to(self.device)

        # Compute action policies
        _, log_probs = self._select_actions(states, action_masks)

        # Compute loss
        # TODO: normalize cumulative_rewards?
        # cumulative_rewards = (cumulative_rewards - cumulative_rewards.mean()) / (cumulative_rewards.std() + eps)
        # TODO: multiply by gamma ** t?
        loss = -log_probs * cumulative_rewards
        loss = loss.sum()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()
