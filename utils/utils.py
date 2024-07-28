from typing import Any

import numpy as np


def get_action_mask(observation: Any, info: Any) -> np.ndarray:
    """
    return the current valid action mask
    :param observation: observation received from environment
    :param info: info received from environment
    :return: a numpy array representing the mask
    """
    if 'action_mask' in info:
        return info['action_mask']
    elif isinstance(observation, dict) and 'action_mask' in observation:
        return observation['action_mask']
    raise ValueError('Could not find action mask')
