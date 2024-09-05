from functools import partial
from pathlib import Path
from typing import Any, Union, Literal, Tuple, Callable

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
        (1, -1): (0.75, 0.15),
        (-1, 1): (0.15, 0.75),
        (1, 0): (0.55, 0.25),
        (0, 1): (0.25, 0.55),
        None: (0.15, 0.15),
    })

    def __init__(self, depth: int = 3, reduction_method: Literal['mean', 'probabilistic'] = 'probabilistic',
                 db_path: Union[Path, str] = UnbeatableClassicTTTAgent.DEFAULT_DB_PATH,
                 depth_zero_probability_estimations: dict = _DEFAULT_DEPTH_ZERO_PROBABILITY_ESTIMATIONS,
                 *args, **kwargs) -> None:
        """
        :param depth: in each sub-board, at what depth to stop calculating (over 9 means iterate until board is full)
        :param reduction_method: how to reduce (aggregate) all the sub-boards probabilities. options:
            probabilistic: approximate the probability the win by approximating the probability for each winning streak
            mean: mean the probabilities over the boards
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
        self._cachable_prob_to_win_sub_board = self._make_cachable(self._prob_to_win_sub_board)
        self._cachable_main_eval = self._make_cachable(self._main_eval)

        assert reduction_method in ['probabilistic', 'mean']

    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        if isinstance(obs, dict):
            obs = obs['observation']

        board = np.where(obs[..., 0], Piece.X.value, 0) + np.where(obs[..., 1], Piece.O.value, 0)
        prob_to_win, prob_to_lose = self._cachable_main_eval(board=board)
        return prob_to_win - prob_to_lose

    def _get_prob_matrix(self, board: np.ndarray) -> np.ndarray:
        prob_matrix = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                sub_board = board[i, j]
                prob_matrix[i, j] = self._cachable_main_eval(board=sub_board)
        return prob_matrix

    def _main_eval(self, board: np.ndarray):
        if board.shape == (3, 3):  # base case
            return self._cachable_prob_to_win_sub_board(board=board, depth=self.depth)

        prob_matrix = self._get_prob_matrix(board)
        prob_to_win, prob_to_lose = self._prob_matrix_to_outer_probs(prob_matrix)
        assert 0 <= prob_to_win <= 1
        assert 0 <= prob_to_lose <= 1
        return prob_to_win, prob_to_lose

    def _cachable_outer_f(self, f: Callable, **kwargs):
        cache_key = []
        for arg in kwargs.values():
            if isinstance(arg, np.ndarray):
                cache_key.append(hashlib.sha1(arg).hexdigest())
            else:
                cache_key.append(arg)
        cache_key = tuple(cache_key)

        if cache_key in self.cache:
            return self.cache[cache_key]

        res = f(**kwargs)
        self.cache[cache_key] = res
        return res

    def _make_cachable(self, f: Callable):
        return partial(self._cachable_outer_f, f=f)

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

    def _prob_matrix_to_outer_probs(self, prob_matrix: np.ndarray) -> Tuple[float, float]:
        """
        :param prob_matrix: a 3x3x2, where cell (i, j, k) is the probability of player k to win in sub-board (i, j)
        :return: the resulting score (depends on the reduction method)
        """
        assert prob_matrix.shape == (3, 3, 2)

        if self.reduction_method == 'mean':
            return prob_matrix[..., 0].mean(), prob_matrix[..., 1].mean()

        if self.reduction_method == 'probabilistic':
            flattened_prob_matrix = prob_matrix.reshape(9, 2)
            prob_per_win_streak = flattened_prob_matrix[self._POSSIBLE_WIN_SEQUENCES].prod(axis=1)
            assert prob_per_win_streak.shape == (len(self._POSSIBLE_WIN_SEQUENCES), 2)
            prob_to_win, prob_to_lose = prob_per_win_streak[:, 0].mean(), prob_per_win_streak[:, 1].mean()
            return prob_to_win, prob_to_lose

        raise ValueError('Unknown reduction method')
