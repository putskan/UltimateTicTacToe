from abc import abstractmethod
from typing import Optional, Dict, Any

from agents.agent import Agent
from utils.replay_buffer import ReplayBuffer


class TrainableAgent(Agent):
    """
    An agent that needs training (neural net, etc.)
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.modules = []  # points to the agent's torch modules. add all modules to this list

    @abstractmethod
    def train_update(self, replay_buffer: ReplayBuffer) -> Optional[Dict[str, Any]]:
        """
        relevant for agents that train (neural networks).
        perform a train step/update and return the loss.
        """
        pass

    def eval(self) -> None:
        """
        change inner models to eval mode (for pyTorch neural nets)
        for example, should include lines such as self.model.train()
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
        """
        assert len(self.modules) > 0, 'All trainable modules should be stored in self.modules'
        for module in self.modules:
            module.eval()

    def train(self) -> None:
        """
        change inner models to train mode (for pyTorch neural nets)
        for example, should include lines such as self.model.train()
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
        """
        assert len(self.modules) > 0, 'All trainable modules should be stored in self.modules'
        for module in self.modules:
            module.train()
