from abc import abstractmethod
from typing import Any, Optional

import numpy as np
from pettingzoo import AECEnv


class Agent:
    """
    base class for all agents
    """
    def __init__(self, agent_name: str = None):
        """
        initialize the agent
        :param agent_name: name to display in str, repr
        """
        self.agent_name = agent_name

    @abstractmethod
    def play(self, env: AECEnv, obs: Any, curr_agent_str: str, action_mask: Optional[np.ndarray]) -> Any:
        """
        :param env: the env (useful to get possible actions, etc.)
        :param obs: current observation
        :param curr_agent_str: parameter of env.action_space. index of the current player
        :param action_mask: mask to apply to the actions, in order to get the valid ones
                (For example, env.action_space(agent).sample(mask))
        :return: the action to play
        """
        pass

    def __str__(self) -> str:
        return self.agent_name or self.__class__.__name__

    def __repr__(self) -> str:
        return self.agent_name or self.__class__.__name__
