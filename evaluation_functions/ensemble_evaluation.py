import math
from typing import Any, Iterable, List

import numpy as np
from pettingzoo import AECEnv
from evaluation_functions.evaluation_function import EvaluationFunction


class EnsembleEvaluation(EvaluationFunction):
    """
    ensemble multiple evaluation functions
    """
    def __init__(self, evaluation_functions: Iterable[EvaluationFunction],
                 weights: List[float] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.evaluation_functions = evaluation_functions
        self.weights = weights
        if weights is not None:
            assert math.isclose(np.sum(weights), 1)

    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        final_score = 0
        for i, eval_function in enumerate(self.evaluation_functions):
            score = eval_function(env, obs, curr_agent_idx, *args, **kwargs)
            weight = 0.5 if self.weights is None else self.weights[i]
            final_score += score * weight
        return final_score
