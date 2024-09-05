from typing import Any

import numpy as np
import torch

from agents.reinforce_agent import ReinforceAgent
from policy_functions.policy_function import PolicyFunction


class ReinforcePolicy(PolicyFunction):
    def __init__(self, policy_agent: ReinforceAgent):
        super().__init__("REINFORCE")
        self.policy_agent = policy_agent

    def __call__(self, state: Any, action_mask: Any, *args, **kwargs) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32, device=self.policy_agent.device).unsqueeze(0)
        action_mask = torch.BoolTensor(action_mask, device=self.policy_agent.device).unsqueeze(0)
        return self.policy_agent.apply_policy_net(state, action_mask).flatten().detach()