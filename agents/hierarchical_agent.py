import copy
from pathlib import Path
from typing import Any, Optional, Union, Dict

import numpy as np
from pettingzoo import AECEnv

from agents.agent import Agent
from agents.unbeatable_classic_ttt_agent.unbeatable_classic_ttt_agent import UnbeatableClassicTTTAgent

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / 'unbeatable_classic_ttt_agent' / 'state_db.json'


class HierarchicalAgent(Agent):
    """
    an agent that chooses a board and delegates the sub-board decision to an unbeatable 3x3 agent
    """
    def __init__(self, db_path: Union[Path, str] = _DEFAULT_DB_PATH, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classic_ttt_agent = UnbeatableClassicTTTAgent(db_path)

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int,
             curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:
        if isinstance(obs, dict):
            obs = obs['observation']

        forced_boards = info.get('forced_boards') or None
        if forced_boards is not None and forced_boards[0] != slice(None):
            # forced to play a certain board
            outer_action = forced_boards[0], forced_boards[1]

        else:
            assert obs.shape[-1] == 2
            board = np.full((3, 3, 2), 0)
            np_all_axis = tuple(range(2, obs.ndim - 1))
            curr_agent_win_mask = np.all(obs[..., 0] != 0, axis=np_all_axis)
            opponent_agent_win_mask = np.all(obs[..., 1] != 0, axis=np_all_axis)
            assert curr_agent_win_mask.shape == (3, 3)  # TODO: remove
            assert opponent_agent_win_mask.shape == (3, 3)  # TODO: remove
            board[curr_agent_win_mask, 0] = 1
            board[opponent_agent_win_mask, 1] = 1
            outer_action_flattened = self.classic_ttt_agent.play(env, board, curr_agent_idx,
                                                                 curr_agent_str, None, info)
            outer_action = np.unravel_index(outer_action_flattened, (3, 3))

        if obs.ndim > 3:
            # recursively get the inner actions
            if forced_boards is not None:
                info = copy.deepcopy(info)
                info['forced_boards'] = forced_boards[2:]
            recursion_action_flattened = self.play(env, obs[*outer_action], curr_agent_idx,
                                                   curr_agent_str, None, info)
            recursion_action = np.unravel_index(recursion_action_flattened, list(obs.shape)[2: -1])
        else:
            recursion_action = tuple()

        flattened_action = np.ravel_multi_index(list(outer_action) + list(recursion_action), list(obs.shape)[:-1])
        return flattened_action
