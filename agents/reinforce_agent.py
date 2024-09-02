import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from agents.trainable_agent import TrainableAgent
from models.dqn import DQN
from models.reinforce import ReinforcePolicy
from utils.replay_buffer import ReplayBuffer


class ReinforceAgent(TrainableAgent):
    """
    A REINFORCE agent that uses a policy network to select actions
    """
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=1e-4, discount_factor=0.99, epsilon=0.7,
                 epsilon_decay=0.99, epsilon_min=0.01, batch_size=64, use_lr_scheduler: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.use_lr_scheduler = use_lr_scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_eps_greedy = False

        # self.policy_net = ReinforcePolicy(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.99999)

        self.modules.append(self.policy_net)

    def eval(self) -> None:
        super().eval()
        self.use_eps_greedy = False

    def train(self) -> None:
        super().train()
        self.use_eps_greedy = True

    def play(self, env, obs, curr_agent_idx, curr_agent_str, action_mask, info: Dict[str, Any]):
        if self.use_eps_greedy and random.random() <= self.epsilon:
            if callable(env.action_space):
                return env.action_space(curr_agent_str).sample(action_mask).item()
            else:
                return env.action_space.sample(action_mask.astype(np.int8)).item()

        state = torch.tensor(obs['observation'], dtype=torch.float32).unsqueeze(0).to(self.device)
        action_mask = torch.BoolTensor(action_mask).to(self.device).unsqueeze(0)
        action = self._select_actions(state, action_mask)
        return action.item()

    def _apply_policy_net(self, states: Tensor, action_masks: Tensor) -> Categorical:
        """
        Apply the policy network to the states and return a Categorical distribution of the actions
        according to the action masks
        :param states: The states to apply the policy network to
        :param action_masks: The action masks that correspond to the states
        :return: a Categorical distribution of the actions corresponding to the states
        """
        policy_vals = self.policy_net(states)
        policy_vals[action_masks.bitwise_not()] = -float('inf')
        probs = nn.functional.softmax(policy_vals, dim=-1)
        m = Categorical(probs)
        return m

    def _select_actions(self, states: Tensor, action_masks: Tensor) -> Tensor:
        """
        Select actions according to the policy network
        :param states: The states to select actions for
        :param action_masks: The action masks that correspond to the states
        :return: The selected actions
        """
        m = self._apply_policy_net(states, action_masks)
        actions = m.sample()
        return actions

    def get_action_log_probs(self, states: Tensor, action_masks: Tensor, actions: Tensor) -> Tensor:
        """
        Get the log probabilities of the actions according to the policy network
        :param states: The states that correspond to the actions
        :param action_masks: The action masks that correspond to the states
        :param actions: The actions to get the log probabilities for
        :return: The log probabilities of the actions
        """
        m = self._apply_policy_net(states, action_masks)
        return m.log_prob(actions)

    def train_update(self, replay_buffer: ReplayBuffer) -> Optional[Dict[str, Any]]:
        if len(replay_buffer) < self.batch_size * 5:
            print("Replay buffer not large enough to train")
            return

        batch = replay_buffer.sample(self.batch_size)

        (states, actions, _, _, _, action_masks,
         _, _, cumulative_rewards, _) = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        action_masks = torch.BoolTensor(action_masks).to(self.device)
        cumulative_rewards = torch.FloatTensor(cumulative_rewards).to(self.device)

        # Compute the log probabilities of action policies
        log_probs = self.get_action_log_probs(states, action_masks, actions)

        # Compute loss
        # TODO: multiply by discount_factor ** t?
        loss = -log_probs * cumulative_rewards
        loss = loss.sum()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        if self.use_lr_scheduler:
            self.lr_scheduler.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {'loss': loss.item(), 'epsilon': self.epsilon, 'lr': self.lr_scheduler.get_last_lr()[0]}
