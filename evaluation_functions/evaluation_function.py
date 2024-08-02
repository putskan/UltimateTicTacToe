from abc import abstractmethod
from typing import Any
from pettingzoo import AECEnv


class EvaluationFunction:
    def __init__(self, name: str = None):
        """
        initialize the evaluation function
        :param name: name to display in str, repr
        """
        self.name = name

    @abstractmethod
    def __call__(self, env: AECEnv, obs: Any, curr_agent_idx: int, *args, **kwargs) -> float:
        """
        :param env: the env (useful to get possible actions, etc.)
        :param obs: current observation
        :param curr_agent_idx: current agent to play
        :return: score of the position
        """
        pass

    def __str__(self) -> str:
        return self.name or self.__class__.__name__

    def __repr__(self) -> str:
        return self.name or self.__class__.__name__
