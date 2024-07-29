import random
from collections import deque, namedtuple
from typing import List, Any

import numpy as np

Record = namedtuple('record', ['observation', 'action', 'reward',
                               'next_observation', 'done', 'action_mask',
                               'curr_player_idx'])


class ReplayBuffer:
    """
    stores previously seen samples
    """
    def __init__(self, size: int = 1000):
        """
        :param size: size of replay buffer (old records are be discarded)
        """
        self.size = size
        self.queue = deque(maxlen=size)

    def push(self, observation: Any, action: Any, reward: float, next_observation: Any, done: bool,
             action_mask: np.ndarray, curr_player_idx: int):
        self.queue.append(Record(observation=observation, action=action, reward=reward,
                                 next_observation=next_observation, done=done, action_mask=action_mask,
                                 curr_player_idx=curr_player_idx))

    def sample(self, num_samples: int) -> List[Record]:
        """
        sample from the replay buffer
        :param num_samples: number of samples to fetch
        :return: list of the sampled records
        """
        return random.sample(self.queue, num_samples)

    def __str__(self) -> str:
        return str(self.queue)

    def __len__(self):
        return len(self.queue)
