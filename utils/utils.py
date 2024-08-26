from typing import Any, List

import numpy as np
import pickle

from utils.piece import Piece


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


def calculate_winning_combinations() -> List[Any]:
    """
    For the classic 3x3 TTT,
    calculate and save all the winning combinations
    """
    winning_combinations = []
    indices = [x for x in range(0, 9)]

    # vertical combinations
    winning_combinations += [
        tuple(indices[i: (i + 3)]) for i in range(0, len(indices), 3)
    ]

    # horizontal combinations
    winning_combinations += [
        tuple(indices[x] for x in range(y, len(indices), 3)) for y in range(0, 3)
    ]

    # diagonal combinations
    winning_combinations.append(tuple(x for x in range(0, len(indices), 4)))
    winning_combinations.append(tuple(x for x in range(2, len(indices) - 1, 2)))

    return winning_combinations


def check_for_winner_classic(three_on_three_board: np.ndarray,
                             winning_combinations: List[Any]) -> int:
    """
    for a 3x3 board, return the winner, or -1 if there's no winner
    :param three_on_three_board: numpy array of shape (3, 3)
    :param winning_combinations: received from calculate_winning_combinations function
    :return: the winner number, -1 otherwise
    """
    assert three_on_three_board.shape == (3, 3)
    flattened_board = three_on_three_board.flatten().tolist()
    winner = -1
    for combination in winning_combinations:
        states = []
        for index in combination:
            states.append(flattened_board[index])
        if all(x == Piece.X.value for x in states):
            winner = Piece.X.value
        if all(x == Piece.O.value for x in states):
            winner = Piece.O.value
    return winner
