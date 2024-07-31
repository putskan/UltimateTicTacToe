from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from pettingzoo import AECEnv

from agents.agent import Agent
from agents.unbeatable_classic_ttt_agent.unbeatable_classic_ttt_agent import UnbeatableClassicTTTAgent
from utils.piece import Piece

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / 'unbeatable_classic_ttt_agent' / 'state_db.json'


class HierarchicalAgent(Agent):
    """
    an agent that chooses a board and delegates the sub-board decision to an unbeatable 3x3 agent
    """
    def __init__(self, db_path: Union[Path, str] = _DEFAULT_DB_PATH, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classic_ttt_agent = UnbeatableClassicTTTAgent(db_path)

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int,
             curr_agent_str: str, action_mask: Optional[np.ndarray]) -> Any:
        if isinstance(obs, dict):
            obs = obs['observation']

        assert obs.shape[-1] == 2
        board = np.full((3, 3, 2), Piece.EMPTY.value)
        np_all_axis = tuple(range(2, obs.ndim - 1))
        curr_agent_win_mask = np.all(obs[..., 0] != 0, axis=np_all_axis)
        opponent_agent_win_mask = np.all(obs[..., 1] != 0, axis=np_all_axis)
        assert curr_agent_win_mask.shape == (3, 3)  # TODO: remove
        assert opponent_agent_win_mask.shape == (3, 3)  # TODO: remove
        board[curr_agent_win_mask, 0] = 1
        board[opponent_agent_win_mask, 1] = 1
        outer_action_flattened = self.classic_ttt_agent.play(env, board, curr_agent_idx, curr_agent_str, None)
        outer_action = np.unravel_index(outer_action_flattened, (3, 3))

        if obs.ndim > 3:
            recursion_action_flattened = self.play(env, obs[*outer_action], curr_agent_idx, curr_agent_str, None)
            recursion_action = np.unravel_index(recursion_action_flattened, list(obs.shape)[2: -1])
        else:
            recursion_action = tuple()
        flattened_action = np.ravel_multi_index(list(outer_action) + list(recursion_action), list(obs.shape)[:-1])
        return flattened_action
