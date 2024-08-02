from typing import Any

import numpy as np
from pettingzoo import AECEnv

from evaluation_functions.evaluation_function import EvaluationFunction


class SubBoardsWon(EvaluationFunction):
    """
    calculate how much sub boards our player won wrt to the other player
    more specifically, score = (curr_player_won_sub_boards - opponent_won_sub_boards)
    the sub-boards in this case are the deepest boards.

    for example, if we play ultimate ttt with depth=3, and our player won the top-left outer board,
    we get score = 9 - 0 = 9, because we essentially won all its 9 sub-boards.

    for example, at the beginning of the game we have score=0
    """
    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        if isinstance(obs, dict):
            obs = obs['observation']

        curr_player_pieces = obs[..., 0]
        opponent_player_pieces = obs[..., 1]

        curr_player_won = np.all(curr_player_pieces == 1, axis=-1).sum()
        opponent_player_won = np.all(opponent_player_pieces == 1, axis=-1).sum()
        return curr_player_won - opponent_player_won
