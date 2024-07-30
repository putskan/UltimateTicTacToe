from typing import Any

import numpy as np
import pickle


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


def save_agent(agent, path):
    """
    save the agent to a file
    :param agent: the agent to save
    :param path: the path to save the agent to
    """
    with open(path, 'wb') as f:
        pickle.dump(agent, f)


def load_agent(path):
    """
    load an agent from a file
    :param path: the path to load the agent from
    :return: the loaded agent
    """
    with open(path, 'rb') as f:
        return pickle.load(f)