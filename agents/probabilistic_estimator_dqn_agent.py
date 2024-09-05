from typing import Dict, Any, Optional, Type, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from pettingzoo import AECEnv
from torch.optim.lr_scheduler import ExponentialLR

from agents.dqn_agent import DQNAgent
from agents.trainable_agent import TrainableAgent
from evaluation_functions.probabilistic_estimator import ProbabilisticEstimator
from models.dqn import DQN
from utils.piece import Piece
from utils.replay_buffer import ReplayBuffer
import copy


class ProbabilisticEstimatorDQNAgent(DQNAgent):
    def __init__(self, pe_depth: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pe = ProbabilisticEstimator(depth=pe_depth)

    def _preprocess_using_pe(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        preprocess the observation using ProbabilisticEstimator
        :param obs: observation to utilize
        :return: a Bx3x3 matrix of probs (B being batch size), or 3x3 in case there's no batch dimension
        """
        board = obs['observation']
        board = np.where(board[..., 0], Piece.X.value, 0) + np.where(board[..., 1], Piece.O.value, 0)
        is_batched = board.ndim % 2 != 0
        if is_batched:
            prob_matrices = []
            batch_size = len(board)
            for i in range(batch_size):
                prob_matrices.append(self.pe._get_prob_matrix(board[i]))
            new_boards = np.array(prob_matrices)
            assert new_boards.shape == (batch_size, 3, 3, 2)
        else:
            new_boards = self.pe._get_prob_matrix(board)
        return {**obs, 'pe_observation': new_boards}

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int,
             curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:
        obs = self._preprocess_using_pe(obs)
        # TODO: consider passing both inputs (it might ot know where to place. perhaps its more like a value function though
        return super().play(env, obs, curr_agent_idx, curr_agent_str, action_mask, info)

    def train_update(self, replay_buffer: ReplayBuffer) -> Optional[Dict[str, Any]]:
        if len(replay_buffer) < self.batch_size * 5:
            return

        batch = replay_buffer.sample(self.batch_size)

        (states, actions, rewards, next_states, dones, action_masks,
         curr_player_idxs, next_action_mask, _, _) = zip(*batch)

        preprocessed_states = self._preprocess_using_pe({'observation': np.array(states)})
        preprocessed_next_states = self._preprocess_using_pe({'observation': np.array(next_states)})
        # TODO: something prettier

        states = torch.FloatTensor(preprocessed_states['observation'], device=self.device)
        pe_states = torch.FloatTensor(preprocessed_states['pe_observation'], device=self.device)
        next_states = torch.FloatTensor(preprocessed_next_states['observation'], device=self.device)
        pe_next_states = torch.FloatTensor(preprocessed_next_states['pe_observation'], device=self.device)
        actions = torch.LongTensor(np.array(actions), device=self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards), device=self.device)
        dones = torch.FloatTensor(np.array(dones), device=self.device)
        action_mask_tensor = torch.FloatTensor(np.array(action_masks), device=self.device)

        # compute Q-values for current states
        q_values = self.policy_net(states, action_mask_tensor, pe_states).gather(1, actions).squeeze(1)

        # compute Q-values for next states using the target network
        with torch.no_grad():
            next_action_mask = np.array(next_action_mask)
            next_action_mask_bool = next_action_mask.astype(bool)
            next_action_mask_tensor = torch.FloatTensor(next_action_mask, device=self.device)
            all_next_q_values = np.ma.masked_array(self.target_net(next_states, next_action_mask_tensor, pe_next_states), ~next_action_mask_bool)
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
