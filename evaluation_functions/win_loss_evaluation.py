from typing import Any

import numpy as np
from pettingzoo import AECEnv
from evaluation_functions.evaluation_function import EvaluationFunction


class WinLossEvaluation(EvaluationFunction):
    """
    +1 score if we won, -1, if we lost, 0 otherwise
    """
    def __init__(self, win_value: float = 1, loss_value: float = -1,
                 draw_value: float = 0, non_terminal_value: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win_value = win_value
        self.loss_value = loss_value
        self.draw_value = draw_value
        self.non_terminal_value = non_terminal_value

    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        if isinstance(obs, dict):
            obs = obs['observation']

        our_pieces = obs[..., 0] != 0
        opponent_pieces = obs[..., 1] != 0
        is_terminal = np.all(np.logical_or(our_pieces, opponent_pieces))
        if is_terminal:
            if np.any(our_pieces) and np.any(opponent_pieces):
                return self.draw_value
            if np.any(our_pieces):
                return self.win_value
            if np.any(opponent_pieces):
                return self.loss_value

        return self.non_terminal_value
