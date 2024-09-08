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
    def __init__(self, dqn_agent: DQNAgent, soft_dqn: bool = False,
                 normalize: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dqn_agent = copy.deepcopy(dqn_agent)
        self.dqn_agent.eval()
        self.soft_dqn = soft_dqn
        self.normalize = normalize

    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        q_values = self.dqn_agent.get_q_values(obs)
        if self.soft_dqn:
            probs = torch.nn.functional.softmax(q_values, dim=-1)
            sampled_idx = Categorical(probs).sample()
            score = q_values[:, sampled_idx]
        else:
            score = q_values.max()

        if self.normalize:
            score = (score / 1.45).clip(-1, 1)
        return score.item()
