from typing import Any

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from policy_functions.policy_function import PolicyFunction


class SoftDQNPolicy(PolicyFunction):
    def __init__(self, policy_agent: DQNAgent):
        super().__init__("Soft DQN")
        self.policy_agent = policy_agent
        self.policy_agent.eval()

    def __call__(self, state: Any, action_mask: Any, *args, **kwargs) -> np.ndarray:
        observation = {
            "observation": state,
            "action_mask": action_mask
        }
        q_values = self.policy_agent.get_q_values(observation)
        probs = torch.nn.functional.softmax(q_values, dim=-1)
        return probs.flatten().detach()
