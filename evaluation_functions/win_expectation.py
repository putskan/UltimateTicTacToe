from pathlib import Path
from typing import Any, Union, Literal, Tuple

import numpy as np
import hashlib

from frozendict import frozendict
from pettingzoo import AECEnv

from agents.unbeatable_classic_ttt_agent.unbeatable_classic_ttt_agent import UnbeatableClassicTTTAgent
from evaluation_functions.evaluation_function import EvaluationFunction
from utils.piece import Piece
from utils.utils import calculate_winning_combinations, check_for_winner_classic


class ProbabilisticEstimator(EvaluationFunction):
    """
    for every sub-board, approximate the probability to win in it
    (assuming in every turn, each player has 50% chance to play in it (bernoulli 0.5))

    using those values, approximate the probability of getting a winning (or losing streak)
    sum those streak probabilities and take diff between win and loss streak values.
    """
    _POSSIBLE_WIN_SEQUENCES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    _DEFAULT_DEPTH_ZERO_PROBABILITY_ESTIMATIONS = frozendict({
        (1, -1): (0.55, 0.25),
        (-1, 1): (0.25, 0.55),
        (1, 0): (0.45, 0.35),
        (0, 1): (0.35, 0.45),
        None: (0.35, 0.35),
    })

    def __init__(self, depth: int = 100, reduction_method: Literal['sum', 'probabilistic'] = 'probabilistic',
                 db_path: Union[Path, str] = UnbeatableClassicTTTAgent.DEFAULT_DB_PATH,
                 depth_zero_probability_estimations: dict = _DEFAULT_DEPTH_ZERO_PROBABILITY_ESTIMATIONS,
                 *args, **kwargs) -> None:
        """
        :param depth: in each sub-board, at what depth to stop calculating (over 9 means iterate until board is full)
        :param reduction_method: how to reduce (aggregate) all the sub-boards probabilities. options:
            probabilistic: approximate the probability the win by approximating the probability for each winning streak
            sum: sum the probabilities over the boards
        :param db_path: db path for the unbeatable agent
        :param args: args for parent class
        :param kwargs: kwargs for parent class
        """
        super().__init__(*args, **kwargs)
        self.classic_ttt_agent = UnbeatableClassicTTTAgent(db_path)
        self.depth = depth
        self.winning_combinations = calculate_winning_combinations()
        self.reduction_method = reduction_method
        self.cache = {}
        self.depth_zero_probability_estimations = depth_zero_probability_estimations

        assert reduction_method in ['probabilistic', 'sum']

    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        if isinstance(obs, dict):
            obs = obs['observation']

        # TODO: support depth > 2, and don't forget 'unavailable' piece (like in hierarchical)
        board = np.where(obs[..., 0], Piece.X.value, 0) + np.where(obs[..., 1], Piece.O.value, 0)
        prob_matrix = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                # TODO: split to tuple in an efficient way
                sub_board = board[i, j]
                prob_matrix[i, j] = self._cachable_prob_to_win_sub_board(sub_board, depth=self.depth)

        score = self._prob_matrix_to_score(prob_matrix)
        return score

    def _cachable_prob_to_win_sub_board(self, board: np.ndarray, depth: int) -> Tuple[float, float]:
        """
        see _prob_to_win_sub_board
        """
        board_hash = hashlib.sha1(board).hexdigest()
        key = board_hash, depth
        if key in self.cache:
            return self.cache[key]

        res = self._prob_to_win_sub_board(board, depth)
        self.cache[key] = res
        return res

    def _prob_to_win_sub_board(self, board: np.ndarray, depth: int) -> Tuple[float, float]:
        """
        go over all trajectories, as a bernoulli process, and calculate the probability for each player to win
        :param board: 3x3 board
        :param depth: current depth
        :return: prob to win, prob to loss (the remaining portion is the draw probability)
        """
        assert board.shape == (3, 3)
        winner = check_for_winner_classic(board, self.winning_combinations)
        if winner == Piece.X.value:
            return 1, 0
        if winner == Piece.O.value:
            return 0, 1
        if np.all(board != 0):  # draw
            return 0, 0

        my_best_action, my_score = self.classic_ttt_agent.get_best_action(board, 0)
        opponent_best_action, opponent_score = self.classic_ttt_agent.get_best_action(board, 1)

        if depth == 0:
            # approximate the chance for each side to win.
            # can't know for sure because a player can have consecutive turns
            # TODO: tune those params
            if (my_score, opponent_score) in self.depth_zero_probability_estimations:
                return self.depth_zero_probability_estimations[(my_score, opponent_score)]
            return self.depth_zero_probability_estimations[None]

        board[*my_best_action] = Piece.X.value
        path_1_win_prob, path_1_loss_prob = self._prob_to_win_sub_board(board, depth - 1)
        board[*my_best_action] = 0
        board[*opponent_best_action] = Piece.O.value
        path_2_win_prob, path_2_loss_prob = self._prob_to_win_sub_board(board, depth - 1)
        board[*opponent_best_action] = 0

        probs = (path_1_win_prob + path_2_win_prob) / 2, (path_1_loss_prob + path_2_loss_prob) / 2
        return probs

    def _prob_matrix_to_score(self, prob_matrix: np.ndarray) -> float:
        """
        :param prob_matrix: a 3x3x2, where cell (i, j, k) is the probability of player k to win in sub-board (i, j)
        :return: the resulting score (depends on the reduction method)
        """
        assert prob_matrix.shape == (3, 3, 2)

        if self.reduction_method == 'sum':
            return (prob_matrix[..., 0] - prob_matrix[..., 1]).sum()

        if self.reduction_method == 'probabilistic':
            flattened_prob_matrix = prob_matrix.reshape(9, 2)
            prob_per_win_streak = flattened_prob_matrix[self._POSSIBLE_WIN_SEQUENCES].prod(axis=1)
            assert prob_per_win_streak.shape == (len(self._POSSIBLE_WIN_SEQUENCES), 2)
            probs_diff = prob_per_win_streak[:, 0].sum() - prob_per_win_streak[:, 1].sum()
            return probs_diff.item()

        raise ValueError('Unknown reduction method')


