import copy
from typing import Any, Optional, Dict, Tuple

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.classic import tictactoe_v3

from agents.agent import Agent
from agents.hierarchical_agent import HierarchicalAgent
from agents.random_agent import RandomAgent
from environments import ultimate_ttt
from evaluate import evaluate_agents
from evaluation_functions.constant_evaluation import ConstantEvaluation
from evaluation_functions.evaluation_function import EvaluationFunction
from evaluation_functions.sub_boards_won import SubBoardsWon
from utils.utils import get_action_mask

DEFAULT_SCORES_ON_GAME_OVER = (float('inf'), 0)  # win, tie


class AlphaBeta(Agent):
    """
    AlphaBeta search, which uses an evaluation function passed in its constructor
    """

    def __init__(self, depth: int, evaluation_function: EvaluationFunction,
                 scores_on_game_over: Optional[Tuple[float, float]] = DEFAULT_SCORES_ON_GAME_OVER,
                 shuffle_move_order: bool = False,
                 *args, **kwargs):
        """

        :param depth: depth to perform the search
        :param evaluation_function: evaluation function to apply when reaching depth=0
        :param scores_on_game_over: on a game-over leaf, what should score (evaluation) should the alpha-beta give
                                    default if inf for win and 0 for draw (and -inf for loss)
                                    if None is passed instead, use the provided evaluation function
                                    for game-over states as well.

                                    if the provided eval function gives positive score for "good" positions and
                                    negative for "bad" positions, you should probably use the default argument.

        :param shuffle_move_order: if True, this is somewhat equivalent to taking a random move if there are
                                    equally good options
        """
        super().__init__(*args, **kwargs)
        if 'agent_name' not in kwargs:
            shuffle_str = ', shuffle_moves' if shuffle_move_order else ''
            self.agent_name = f'{self.__class__.__name__} (Depth {depth}, Eval Func {evaluation_function}{shuffle_str})'
        self.depth = depth
        self.evaluation_function = evaluation_function
        self.scores_on_game_over = scores_on_game_over
        self.shuffle_move_order = shuffle_move_order

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int, curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:
        score, action = self._alpha_beta(env, self.depth, alpha=-np.inf, beta=np.inf,
                                         maximizing_player=True, curr_agent_idx=curr_agent_idx)
        return action

    @staticmethod
    def _deepcopy_env(env: AECEnv) -> AECEnv:
        """
        copy all attributes but 'board_renderer' (not pickleable)
        :param env: environment to copy
        :return: a deep copy of the environment
        """
        unwrapped_env = getattr(env, 'unwrapped', env)
        board_renderer = getattr(unwrapped_env, 'board_renderer', None)
        if board_renderer is not None:
            unwrapped_env.board_renderer = None

        copied_env = copy.deepcopy(env)

        if board_renderer is not None:
            unwrapped_env.board_renderer = board_renderer

        return copied_env

    def _alpha_beta(self, env: AECEnv, curr_depth: int, alpha: float, beta: float,
                    maximizing_player: bool, curr_agent_idx: int) -> Tuple[float, Any]:
        obs, reward, termination, truncation, info = env.last()
        action_mask = get_action_mask(obs, info)

        done = termination or truncation
        if done and self.scores_on_game_over is not None:
            if reward not in [0, 1, -1]:
                raise ValueError(f"Unexpected reward {reward}")

            if reward == 0:  # draw
                return self.scores_on_game_over[1], np.argmax(action_mask)

            maximizer_won_sign = reward * (maximizing_player * 2 - 1)
            return maximizer_won_sign * self.scores_on_game_over[0], np.argmax(action_mask)

        if curr_depth == 0 or done:
            return self.evaluation_function(env, obs, curr_agent_idx), np.argmax(action_mask)

        valid_actions = np.where(action_mask)[0]
        if self.shuffle_move_order:
            np.random.shuffle(valid_actions)

        if maximizing_player:
            max_eval = -np.inf
            max_action = None
            for action in valid_actions.tolist():
                curr_env = self._deepcopy_env(env)
                curr_env.step(action)
                score, _ = self._alpha_beta(curr_env, curr_depth - 1, alpha, beta, False, 1 - curr_agent_idx)
                if score >= max_eval:
                    max_eval = score
                    max_action = action
                alpha = max(alpha, score)
                if beta <= alpha:
                    # cutoff
                    return float('inf'), action
            return max_eval, max_action

        else:
            min_eval = np.inf
            min_action = None
            for action in valid_actions.tolist():
                curr_env = copy.deepcopy(env)
                curr_env.step(action)
                score, _ = self._alpha_beta(curr_env, curr_depth - 1, alpha, beta, True, 1 - curr_agent_idx)
                if score <= min_eval:
                    min_eval = score
                    min_action = action
                beta = min(beta, score)
                if beta <= alpha:
                    # cutoff
                    return float('-inf'), action
            return min_eval, min_action


if __name__ == '__main__':
    env = ultimate_ttt.env(render_mode=None, depth=2)
    agents = [
        AlphaBeta(depth=1, evaluation_function=ConstantEvaluation(), shuffle_move_order=True),
        AlphaBeta(depth=1, evaluation_function=SubBoardsWon(), shuffle_move_order=True),
        HierarchicalAgent(),
    ]
    evaluate_agents.evaluate_agents(env, agents=agents, n_games=20)
