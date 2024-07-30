import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from agents.trainable_agent import TrainableAgent
from utils.replay_buffer import ReplayBuffer
import torch.nn.functional as F
import copy


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        n_observations = np.prod(n_observations)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent(TrainableAgent):
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001, gamma=0.99, epsilon=0.1,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, replay_buffer_size=10000, target_update_freq=10):
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
        self.target_update_freq = target_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_count = 0  # Counter to track when to update the target network

        # Q-Networks
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.modules.append(self.policy_net)

    def play(self, env, obs, curr_agent_idx, curr_agent_str, action_mask):
        if random.random() <= self.epsilon:
            return env.action_space(curr_agent_str).sample(action_mask)

        state = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy_net(state)
            action[:, (~action_mask.astype(bool)).tolist()] = -float('inf')
            return action.argmax().item()

    def train_update(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return

        batch = replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones, action_masks, curr_player_idxs = zip(*batch)

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
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        print(f"loss - ", loss.item())

        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(copy.deepcopy(self.policy_net.state_dict()))
