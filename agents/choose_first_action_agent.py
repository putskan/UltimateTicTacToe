from typing import Any, Optional

import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv

from agents.agent import Agent


class ChooseFirstActionAgent(Agent):
    """
    Always chooses the first valid action in the list
    """
    def play(self, env: AECEnv, obs: Any, curr_agent_str: str, action_mask: Optional[np.ndarray]) -> Any:
        assert isinstance(env.action_space(curr_agent_str), Discrete)
        action = np.argmax(action_mask).item()  # get first True value
        return action
