from abc import abstractmethod
from typing import Any, Optional

import numpy as np
from pettingzoo import AECEnv


class Agent:
    @abstractmethod
    def play(self, env: AECEnv, obs: Any, curr_agent_str: str, action_mask: Optional[np.ndarray]) -> Any:
        """
        :param env: the env (useful to get possible actions etc)
        :param obs: current observation
        :param curr_agent_str: parameter of env.action_space. index of the current player
        :param action_mask: mask to apply to the actions, in order to get the valid ones
                (For example, env.action_space(agent).sample(mask))
        :return: the action to play
        """
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__
