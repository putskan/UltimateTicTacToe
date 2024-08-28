from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from agents.trainable_agent import TrainableAgent
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer
import copy


class DQNAgent(TrainableAgent):
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=1e-4, gamma=0.99, epsilon=0.1,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64,
                 replay_buffer_size=10000, tau=0.005):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(copy.deepcopy(self.policy_net.state_dict()))
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.modules.append(self.policy_net)

    def play(self, env, obs, curr_agent_idx, curr_agent_str, action_mask, info: Dict[str, Any]):
        if random.random() <= self.epsilon:
            return env.action_space(curr_agent_str).sample(action_mask).item()

        state = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy_net(state)
            action[:, (~action_mask.astype(bool)).tolist()] = -float('inf')
            return action.argmax().item()

    def train_update(self, replay_buffer: ReplayBuffer):
        if len(replay_buffer) < self.batch_size:
            return

        batch = replay_buffer.sample(self.batch_size)

        (states, actions, rewards, next_states, dones, action_masks,
         curr_player_idxs, next_action_mask, _) = zip(*batch)

        states = torch.stack([torch.FloatTensor(state['observation']) for state in states]).to(self.device)
        next_states = torch.stack([torch.FloatTensor(next_state['observation']) for next_state in next_states]).to(
            self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values for current states
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Compute Q-values for next states using the target network
        with torch.no_grad():
            next_action_mask_bool = np.array(next_action_mask).astype(bool)
            all_next_q_values = np.ma.masked_array(self.target_net(next_states), next_action_mask_bool)
            next_q_values = all_next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau +
                                          target_net_state_dict[key] * (1 - self.tau))
        self.target_net.load_state_dict(target_net_state_dict)
