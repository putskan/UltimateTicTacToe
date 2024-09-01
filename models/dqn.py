from typing import Sequence, Tuple, Union

import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class PrevDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
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


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        n_observations = np.prod(n_observations).item() // 2  # merge planes
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(n_observations, max(32, n_observations * 2)),
            nn.ReLU(),
            nn.Linear(max(32, n_observations * 2), 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        x = x[..., 0] - x[..., 1]
        x = self.flatten(x)
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self, hidden_dim: Union[Tuple[int, ...], int], out_shape: Sequence,
                 dropout_p: float = 0, device: torch.device = None):
        super().__init__()
        hidden_dim = np.prod(hidden_dim).item()
        self.flatten = nn.Flatten()
        self.out_shape = out_shape
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout1d(p=dropout_p),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout1d(p=dropout_p),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout1d(p=dropout_p),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, 1, device=device),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(64, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, np.prod(out_shape).item(), device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        assumes no channel dimension. (B, H, W) or (H, W)
        """
        x = self.flatten(x)
        x = self.backbone(x)
        value_head_res = self.value_head(x)
        advantage_head_res = self.advantage_head(x)
        x = value_head_res + advantage_head_res - torch.mean(advantage_head_res, dim=-1, keepdim=True)
        x = x.reshape(-1, self.out_shape)
        return x
