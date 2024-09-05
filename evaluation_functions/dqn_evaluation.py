import copy
from typing import Any

import torch.nn.functional
from pettingzoo import AECEnv
from torch.distributions import Categorical

from agents.dqn_agent import DQNAgent
from evaluation_functions.evaluation_function import EvaluationFunction


class DQNEvaluation(EvaluationFunction):
    """
    use a dqn agent as an evaluation function (essentially get the current state value)
    """
    def __init__(self, dqn_agent: DQNAgent, soft_dqn: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dqn_agent = copy.deepcopy(dqn_agent)
        self.dqn_agent.eval()
        self.soft_dqn = soft_dqn

    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        q_values = self.dqn_agent.get_q_values(obs)
        if self.soft_dqn:
            probs = torch.nn.functional.softmax(q_values, dim=-1)
            sampled_idx = Categorical(probs).sample()
            return q_values[:, sampled_idx].item()

        return q_values.clip(-1, 1).max().item()
