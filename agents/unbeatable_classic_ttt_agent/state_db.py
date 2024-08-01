import json
from typing import Tuple, Dict, List, Any

import numpy as np
from tqdm import tqdm

from utils.piece import Piece
from utils.utils import calculate_winning_combinations, check_for_winner_classic

Database = Dict[str, Tuple[Tuple[int, int], int]]
Action = Tuple[int, int]


def hash_position(board: np.ndarray, player_idx: int) -> str:
    return 'board=' + str(board) + ', player_idx=' + str(player_idx)


def get_best_action(position: np.ndarray, player_idx: int, database: Database,
                    winning_combinations: List[Any] = None) -> Tuple[Action, int]:
    """
    get (or calculate) the best action in a position, given the database
    :param position: the board state
    :param player_idx: index of the player that should move (0/1)
    :param database: db that maps positions to actions and scores
    :param winning_combinations: needed for database creation, but not for inference (if the db is already full)
    :return: the optimal action and its score (1=win, 0=tie, -1=lose)
    """
    player_idx_piece = Piece.X.value if player_idx == 0 else Piece.O.value
    hashed_position = hash_position(position, player_idx)
    if hashed_position in database:
        return database[hashed_position]

    assert winning_combinations is not None, 'Database is not full, must provide winning_combinations'
    position_wo_unavailable = np.where(position != Piece.UNAVAILABLE.value, position, Piece.EMPTY.value)
    winner = check_for_winner_classic(position_wo_unavailable, winning_combinations)
    if winner != -1:
        score = 1 if winner == player_idx_piece else -1
        return (0, 0), score

    empty_cells = np.where(position == Piece.EMPTY.value)
    total_empty_cells = len(empty_cells[0])
    if total_empty_cells == 0 and winner == -1:
        # tie
        return (0, 0), 0

    best_action = None
    best_action_score = None
    for i in range(total_empty_cells):
        curr_action = empty_cells[0][i], empty_cells[1][i]
        position[curr_action] = player_idx_piece

        _, curr_score = database[hash_position(position, 1 - player_idx)]
        curr_score = -curr_score
        if best_action is None or curr_score > best_action_score:
            best_action = curr_action[0].item(), curr_action[1].item()
            best_action_score = curr_score

        position[curr_action] = Piece.EMPTY.value

    return best_action, best_action_score


def create_state_database(use_unavailable_piece_type: bool = True) -> Database:
    """
    Using dynamic programming, goes over all possible (and not possible) positions,
    and finds the best move save them to dict of {state_hash: Tuple[best_move, best_move_score]}
    where state_hash is a hash created by the position and the player turn.
    best_move_score is wrt the current agent (1 if it is winning, -1 losing, 0 draw)

    :param use_unavailable_piece_type: pass True to use as a component of an ultimate-TTT agent.
                                        (this is needed for sub-boards that ended in a draw)
                                        For classic 3x3, pass False
    :return: the database dictionary
    """
    database = {}
    winning_combinations = calculate_winning_combinations()
    pieces = [Piece.EMPTY.value, Piece.X.value, Piece.O.value]
    if use_unavailable_piece_type:
        pieces.append(Piece.UNAVAILABLE.value)
    mesh = np.array(np.meshgrid(pieces, pieces, pieces, pieces, pieces, pieces, pieces, pieces, pieces))
    all_positions = mesh.T.reshape(-1, 3, 3)
    assert all_positions.shape == (len(pieces) ** 9, 3, 3)

    for free_cells in tqdm(range(0, 10)):
        curr_positions = all_positions[np.sum(all_positions == Piece.EMPTY.value, axis=(1, 2)) == free_cells]
        for i in range(len(curr_positions)):
            position = curr_positions[i]
            for player_idx in [0, 1]:
                state_hash = hash_position(position, player_idx)
                best_action = get_best_action(position, player_idx, database, winning_combinations)
                database[state_hash] = best_action

    return database


if __name__ == '__main__':
    # create the state db for the agent
    dst_path = './state_db.json'
    database = create_state_database()
    with open(dst_path, 'w') as f:
        json.dump(database, f)

    # to load it back use:
    with open(dst_path, 'r') as f:
        loaded_database = json.load(f)
