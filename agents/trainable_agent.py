from abc import abstractmethod
from typing import Any

from agents.agent import Agent


class TrainableAgent(Agent):
    """
    An agent that needs training (neural net, etc.)
    """
    @abstractmethod
    def train_update(self, replay_buffer: Any) -> None:
        """
        relevant for agents that train (neural networks).
        perform a train step/update.
        """
        pass

    def eval(self) -> None:
        """
        change inner models to eval mode (for pyTorch neural nets)
        for example, should include lines such as self.model.train()
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval

        """
        pass

    def train(self) -> None:
        """
        change inner models to train mode (for pyTorch neural nets)
        for example, should include lines such as self.model.train()
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
        """
        pass
