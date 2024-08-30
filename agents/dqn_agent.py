from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch.optim.lr_scheduler import ExponentialLR

from agents.trainable_agent import TrainableAgent
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer
import copy


class DQNAgent(TrainableAgent):
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=1e-3, gamma=0.6, epsilon=0.4,
                 epsilon_decay=0.9999, epsilon_min=0.1, batch_size=128,
                 tau=0.005, use_lr_scheduler: bool = False):
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
        self.tau = tau
        self.use_lr_scheduler = use_lr_scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(copy.deepcopy(self.policy_net.state_dict()))
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.99999)

        self.modules.append(self.policy_net)
        self.modules.append(self.target_net)

    def play(self, env, obs, curr_agent_idx, curr_agent_str, action_mask, info: Dict[str, Any]):
        if random.random() <= self.epsilon:
            # todo - remove the change
            return env.action_space(curr_agent_str).sample(action_mask).item()
            # return env.action_space.sample(action_mask.astype(np.int8)).item()

        state = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy_net(state)
            action[:, (~action_mask.astype(bool)).tolist()] = -float('inf')
            return action.argmax().item()

    def train_update(self, replay_buffer: ReplayBuffer) -> Optional[Dict[str, Any]]:
        # TODO: add the return type hinting to other agents as well, and type hinting in general
        if len(replay_buffer) < self.batch_size * 5:
            return

        batch = replay_buffer.sample(self.batch_size)

        (states, actions, rewards, next_states, dones, action_masks,
         curr_player_idxs, next_action_mask, _, _) = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # compute Q-values for current states
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # compute Q-values for next states using the target network
        with torch.no_grad():
            next_action_mask_bool = np.array(next_action_mask).astype(bool)
            all_next_q_values = np.ma.masked_array(self.target_net(next_states), ~next_action_mask_bool)
            next_q_values = torch.tensor(all_next_q_values.max(1))
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau +
                                          target_net_state_dict[key] * (1 - self.tau))
        self.target_net.load_state_dict(target_net_state_dict)
        if self.use_lr_scheduler:
            self.lr_scheduler.step()
        return {'loss': loss.item(), 'epsilon': self.epsilon, 'lr': self.lr_scheduler.get_last_lr()[0]}
