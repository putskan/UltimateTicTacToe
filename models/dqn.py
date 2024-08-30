from typing import Sequence, Tuple, Union

import numpy as np
from torch import nn
import torch


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        n_observations = 9  # TODO: change, TODO: add prev version
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
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
