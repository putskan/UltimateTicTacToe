from typing import Any

from agents.random_agent import RandomAgent
from agents.trainable_agent import TrainableAgent
from utils.replay_buffer import ReplayBuffer


class DummyTrainableAgent(TrainableAgent):
    """
    for testing purposes
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_agent = RandomAgent()

    def train_update(self, replay_buffer: ReplayBuffer) -> None:
        pass

    def play(self, *args, **kwargs) -> Any:
        return self.inner_agent.play(*args, **kwargs)
