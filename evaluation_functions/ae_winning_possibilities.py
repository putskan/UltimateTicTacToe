from typing import Any

import numpy as np
from pettingzoo import AECEnv

from environments.board import Board
from evaluation_functions.evaluation_function import EvaluationFunction
from evaluation_functions.win_loss_evaluation import WinLossEvaluation


class AEWinningPossibilities(EvaluationFunction):
    """
    Implementation of 2013 AE heuristic - 'WinningPossibilities'
    https://www.cs.huji.ac.il/course/2021/ai/projects/2013/U2T3P/files/AI_Report.pdf

    This currently supports depth=2
    """
    _WIN_SCORE = 10 ** 6
    _APPROXIMATE_WIN_SCORE = 7
    _NON_TERMINAL_VALUE = 999
    _CURR_PLAYER_VALUE = 1
    _OPPONENT_PLAYER_VALUE = -1
    _POSSIBLE_WIN_SEQUENCES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    _BIG_BOARD_WEIGHT = 23

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._win_loss_evaluation_function = WinLossEvaluation(non_terminal_value=self._NON_TERMINAL_VALUE)

    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        if isinstance(obs, dict):
            obs = obs['observation']

        win_loss_score = self._win_loss_evaluation_function(env, obs, curr_agent_idx)
        if win_loss_score == 0:
            # draw
            return 0

        if win_loss_score != self._NON_TERMINAL_VALUE:
            # won / lost
            # adding free_cells is a strange choice by them
            free_cells = obs.sum(axis=-1) == 0
            return self._WIN_SCORE + free_cells if win_loss_score == 1 else -self._WIN_SCORE - free_cells

        obs_reduced = (np.where(obs[..., 0], self._CURR_PLAYER_VALUE, 0) +
                       np.where(obs[..., 1], self._OPPONENT_PLAYER_VALUE, 0))
        board_as_mini = Board.convert_board_to_3x3(obs_reduced)
        ret = self._assess_sub_board(board_as_mini) * self._BIG_BOARD_WEIGHT
        for i in range(3):
            for j in range(3):
                sub_board = obs_reduced[i, j]
                if np.all(sub_board == 0):
                    ret += self._assess_sub_board(sub_board)
        return ret

    def _assess_sub_board(self, sub_board: np.ndarray) -> float:
        assert sub_board.shape == (3, 3), 'does not support depth > 2'
        if np.all(sub_board != 0):
            # strange choice by them. what if won/lost the sub-board?
            return 0

        player_counter = 0
        opponent_counter = 0
        sub_board_as_list = sub_board.flatten()
        for seq in self._POSSIBLE_WIN_SEQUENCES:
            filtered_seq = [sub_board_as_list[index] for index in seq if sub_board_as_list[index] != 0]
            if self._CURR_PLAYER_VALUE in filtered_seq:
                if self._OPPONENT_PLAYER_VALUE in filtered_seq:
                    continue
                if len(filtered_seq) > 1:
                    player_counter += self._APPROXIMATE_WIN_SCORE
                player_counter += 1
            elif self._OPPONENT_PLAYER_VALUE in filtered_seq:
                if len(filtered_seq) > 1:
                    opponent_counter += self._APPROXIMATE_WIN_SCORE
                opponent_counter += 1
        return player_counter - opponent_counter
