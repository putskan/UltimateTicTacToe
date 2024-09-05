from abc import abstractmethod
from typing import Any

import numpy as np
from pettingzoo import AECEnv


class PolicyFunction:
    def __init__(self, name: str = None):
        """
        initialize the policy function
        :param name: name to display in str, repr
        """
        self.name = name

    @abstractmethod
    def __call__(self, state: Any, action_mask: Any, *args, **kwargs) -> np.ndarray:
        """
        :param env: the env (useful to get possible actions, etc.)
        :param obs: current observation
        :return: a policy for the current observation
        """
        pass

    def __str__(self) -> str:
        return self.name or self.__class__.__name__

    def __repr__(self) -> str:
        return self.name or self.__class__.__name__