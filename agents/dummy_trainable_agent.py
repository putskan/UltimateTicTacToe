from typing import Any

import numpy as np
from torch import nn

from agents.random_agent import RandomAgent
from agents.trainable_agent import TrainableAgent
from utils.replay_buffer import ReplayBuffer


class DummyTrainableAgent(TrainableAgent):
    """
    for testing and demonstration purposes
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = 0
        self.inner_agent = RandomAgent()
        self.modules.append(nn.Linear(1, 1))

    def train_update(self, replay_buffer: ReplayBuffer) -> None:
        if self.i > 100:
            assert np.any([item.done for item in replay_buffer.queue])
        self.i += 1

    def play(self, *args, **kwargs) -> Any:
        return self.inner_agent.play(*args, **kwargs)
