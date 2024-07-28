from typing import Any, Optional

import numpy as np
from pettingzoo import AECEnv

from agents.agent import Agent


class RandomAgent(Agent):
    """
    chooses an action randomly from the all valid actions
    """
    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int,
             curr_agent_str: str, action_mask: Optional[np.ndarray]) -> Any:
        return env.action_space(curr_agent_str).sample(action_mask)
