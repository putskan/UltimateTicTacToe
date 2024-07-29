from typing import List, Any

import numpy as np


class Board:
    def __init__(self, depth: int):
        self.depth = depth
        self.board_shape = tuple([3, 3] * depth)
        self.squares = np.zeros(self.board_shape)
        self.winning_combinations = self.calculate_winning_combinations()

    def setup(self):
        self.winning_combinations = self.calculate_winning_combinations()

    def get_sub_board_winner(self, sub_board: np.ndarray) -> int:
        """
        given some sub-board (not necessary the full board),
        calculate if it has a winner
        assumes each sub-sub-board is full of player i's symbol if player i won it
        :param sub_board: board of shape (3, ..., 3) with even number of threes
        :return:
        """
        assert sub_board.ndim % 2 == 0
        if sub_board.ndim == 2:
            sub_board = sub_board[..., np.newaxis]

        np_all_axis = tuple(range(2, sub_board.ndim))
        player_1_win_mask = np.all(sub_board == 1, axis=np_all_axis)
        player_2_win_mask = np.all(sub_board == 2, axis=np_all_axis)
        classic_board = np.zeros((3, 3))
        classic_board[player_1_win_mask] = 1
        classic_board[player_2_win_mask] = 2
        classic_board_winner = self.check_for_winner_classic(classic_board)
        return classic_board_winner

    def play_turn(self, agent, pos):
        """
        play the turn, and set the sub-board to the agent's value if it won
        :param agent:
        :param pos:
        :return:
        """
        # if spot is empty
        # if self.squares[pos] != 0:
        #     return
        if agent == 0:
            self.squares[pos] = 1
        elif agent == 1:
            self.squares[pos] = 2

        for i in range(1, self.depth):
            sub_board = self.squares[pos[:-i * 2]]
            sub_board_winner = self.get_sub_board_winner(sub_board)
            if sub_board_winner == -1:
                break
            else:
                self.squares[pos[:-i * 2]] = sub_board_winner

    @staticmethod
    def calculate_winning_combinations() -> List[Any]:
        """
        calculate and save all the winning combinations
        """
        winning_combinations = []
        indices = [x for x in range(0, 9)]

        # Vertical combinations
        winning_combinations += [
            tuple(indices[i : (i + 3)]) for i in range(0, len(indices), 3)
        ]

        # Horizontal combinations
        winning_combinations += [
            tuple(indices[x] for x in range(y, len(indices), 3)) for y in range(0, 3)
        ]

        # Diagonal combinations
        winning_combinations.append(tuple(x for x in range(0, len(indices), 4)))
        winning_combinations.append(tuple(x for x in range(2, len(indices) - 1, 2)))

        return winning_combinations

    def check_for_winner_classic(self, three_on_three_board: np.ndarray) -> int:
        """
        for a 3x3 board, return the winner, or -1 if there's no winner
        :param three_on_three_board: numpy array of shape (3, 3)
        :return: the winner number
        """
        assert three_on_three_board.shape == (3, 3)
        winner = -1
        for combination in self.winning_combinations:
            states = []
            for index in combination:
                states.append(three_on_three_board[index])
            if all(x == 1 for x in states):
                winner = 1
            if all(x == 2 for x in states):
                winner = 2
        return winner

    def check_for_winner(self) -> int:
        """
        if someone won, return its id, or -1 otherwise
        :return:
        """
        return self.get_sub_board_winner(self.squares)

        # winner = -1
        # for combination in self.winning_combinations:
        #     states = []
        #     for index in combination:
        #         states.append(self.squares[index])
        #     if all(x == 1 for x in states):
        #         winner = 1
        #     if all(x == 2 for x in states):
        #         winner = 2
        # return winner

    def check_game_over(self) -> bool:
        """
        :return: True if game is over, False otherwise
        """
        winner = self.check_for_winner()
        if winner in [1, 2]:
            return True
        elif np.all(self.squares != 0):
            # tie
            return True
        else:
            return False

    def __str__(self):
        return str(self.squares)
