from typing import Any
from pettingzoo import AECEnv
from evaluation_functions.evaluation_function import EvaluationFunction


class ConstantEvaluation(EvaluationFunction):
    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        return 0.
