from typing import Any, Optional

import numpy as np


def get_action_mask(observation: Any, info: Any) -> np.ndarray:
    if 'action_mask' in info:
        return info['action_mask']
    elif isinstance(observation, dict) and 'action_mask' in observation:
        return observation['action_mask']
    raise ValueError('Could not find action mask')
