import json
from pathlib import Path
from typing import Any, Optional, Union, Dict

import numpy as np
from pettingzoo import AECEnv

from agents.unbeatable_classic_ttt_agent.state_db import get_best_action
from agents.agent import Agent
from utils.piece import Piece


class UnbeatableClassicTTTAgent(Agent):
    """
    an agent that always wins the classic 3x3 tic-tac-toe
    uses a db of all possible positions
    """
    DEFAULT_DB_PATH = Path(__file__).resolve().parent / 'state_db.json'

    def __init__(self, db_path: Union[Path, str] = DEFAULT_DB_PATH, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(db_path, 'rb') as f:
            self.db = json.load(f)

    def get_best_action(self, position: np.ndarray, player_idx: int):
        """
        get best_action, score
        :param position: current board position
        :param player_idx: player idx to play
        :return:
        """
        return get_best_action(position, player_idx, self.db)

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int,
             curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:
        if isinstance(obs, dict):
            obs = obs['observation']

        # alter the board to the expected representation
        board = np.full((3, 3), Piece.EMPTY.value)
        if curr_agent_idx == 0:
            player_piece = Piece.X.value
            opponent_piece = Piece.O.value
        else:
            player_piece = Piece.O.value
            opponent_piece = Piece.X.value

        board[obs[..., 0] != 0] = player_piece
        board[obs[..., 1] != 0] = opponent_piece
        action, _ = self.get_best_action(board, curr_agent_idx)
        flattened_action = action[0] * 3 + action[1]
        return flattened_action
