import time
from typing import Any, Dict, Optional

import numpy as np
from pettingzoo import AECEnv

from agents.agent import Agent


class TimedAgentWrapper:
    """Wrapper for an agent that keeps track of the average time it takes to play a move"""
    def __init__(self, agent: Agent):
        """
        :param agent: the agent to wrap
        """
        self.agent = agent
        self.num_plays = 0
        self.avg_time = 0

    def update_avg_time(self, elapsed_time: float) -> None:
        """
        Update the average time it takes to play a move
        :param elapsed_time: time it took to play a move
        """
        self.num_plays += 1
        self.avg_time = (self.avg_time * (self.num_plays - 1) + elapsed_time) / self.num_plays

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int,
             curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:
        start_time = time.time()
        action = self.agent.play(env, obs, curr_agent_idx, curr_agent_str, action_mask, info)
        end_time = time.time()
        self.update_avg_time(end_time - start_time)
        return action

    @property
    def average_time(self):
        """Return the average time it takes to play a move"""
        return self.avg_time

    def __getattr__(self, attr):
        return getattr(self.agent, attr)

    def __str__(self):
        return str(self.agent)

    def __repr__(self):
        return repr(self.agent)
