from typing import Dict, Any, Optional, Type, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from pettingzoo import AECEnv
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ExponentialLR

from agents.trainable_agent import TrainableAgent
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer
import copy


class DQNAgent(TrainableAgent):
    def __init__(self, state_size: Union[Tuple[int], int], action_size: int,
                 learning_rate: float = 1e-3, discount_factor: float = 0.6,
                 epsilon: float = 0.4, epsilon_decay: float = 0.9999, epsilon_min: float = 0.1,
                 batch_size: int = 128, tau: float = 0.005, use_lr_scheduler: bool = False, model_cls: Type = DQN,
                 min_lr: float = 5e-6, soft_play: bool = False):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.tau = tau
        self.use_lr_scheduler = use_lr_scheduler
        self.min_lr = min_lr
        self.soft_play = soft_play,
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.policy_net = model_cls(self.state_size, self.action_size).to(self.device)
        self.target_net = model_cls(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(copy.deepcopy(self.policy_net.state_dict()))
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.99999)

        self.modules.append(self.policy_net)
        self.modules.append(self.target_net)

        self.use_eps_greedy = False

    def set_soft_play(self, soft_play: bool) -> None:
        self.soft_play = soft_play

    def copy_networks(self, other: 'DQNAgent'):
        self.policy_net.load_state_dict(other.policy_net.state_dict())
        self.target_net.load_state_dict(other.target_net.state_dict())

    def eval(self) -> None:
        super().eval()
        self.use_eps_greedy = False

    def train(self) -> None:
        super().train()
        self.use_eps_greedy = True

    def get_q_values(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        state = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)
        action_mask = obs['action_mask']
        with torch.no_grad():
            action = self.policy_net(state)
        action[:, (~action_mask.astype(bool)).tolist()] = -float('inf')
        return action

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int,
             curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:
        if self.use_eps_greedy and random.random() <= self.epsilon:
            return env.action_space(curr_agent_str).sample(action_mask).item()

        q_values = self.get_q_values(obs)
        if self.soft_play:
            probs = torch.nn.functional.softmax(q_values, dim=-1)
            sampled_idx = Categorical(probs).sample().item()
            return sampled_idx

        return q_values.argmax().item()

    def train_update(self, replay_buffer: ReplayBuffer) -> Optional[Dict[str, Any]]:
        if len(replay_buffer) < self.batch_size * 5:
            return

        batch = replay_buffer.sample(self.batch_size)

        (states, actions, rewards, next_states, dones, action_masks,
         curr_player_idxs, next_action_mask, _, _) = zip(*batch)

        states = torch.FloatTensor(np.array(states), device=self.device)
        next_states = torch.FloatTensor(np.array(next_states), device=self.device)
        actions = torch.LongTensor(np.array(actions), device=self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards), device=self.device)
        dones = torch.FloatTensor(np.array(dones), device=self.device)

        # compute Q-values for current states
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # compute Q-values for next states using the target network
        with torch.no_grad():
            next_action_mask_bool = np.array(next_action_mask).astype(bool)
            all_next_q_values = np.ma.masked_array(self.target_net(next_states), ~next_action_mask_bool)
            next_q_values = torch.tensor(all_next_q_values.max(1))
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

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
        if self.use_lr_scheduler and self.lr_scheduler.get_last_lr()[0] > self.min_lr:
            self.lr_scheduler.step()
        return {'loss': loss.item(), 'epsilon': self.epsilon, 'lr': self.lr_scheduler.get_last_lr()[0]}
