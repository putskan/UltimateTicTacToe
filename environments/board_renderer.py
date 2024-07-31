from enum import Enum
from typing import Tuple

import numpy as np
import pygame


class Piece(Enum):
    EMPTY = 0
    X = 1
    O = 2


class BoardRenderer:
    """
    render an ultimate TTT board
    """
    # color palette attribute goes to AlejoG10
    BG_COLOR = (28, 170, 156)
    LINE_COLOR = (23, 145, 135)
    CROSS_COLOR = (66, 66, 66)
    CIRCLE_COLOR = (239, 231, 200)
    PADDING = 20
    LINE_WIDTH = 10

    def __init__(self, caption: str = 'Ultimate Tic-Tac-Toe', width: int = 729,
                 height: int = 729, render_fps: int = 1) -> None:
        self.caption = caption
        self.width = width
        self.height = height
        self.render_fps = render_fps

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

    @staticmethod
    def draw_sub_board(screen: pygame.Surface,
                       sub_board: np.ndarray,
                       top_left: Tuple[float, float],
                       bottom_right: Tuple[float, float],
                       line_width: int, padding: int) -> None:
        """
        draw the sub board's 4 lines, and call recursion for the inner boards
        :param screen: screen to draw on
        :param sub_board: numpy array of even number of dimensions (3, ..., 3)
        :param top_left: top left position available for the sub-board (e.g., in the original board its (0, 0))
        :param bottom_right: bottom right position available for the sub-board (e.g., in the original board its (W, H))
        :param line_width: line width
        :param padding: padding for each sub-board
        NOTE: in pygame the coords are (W, H), and top-left is (0, 0)
        """
        assert sub_board.ndim % 2 == 0
        if sub_board.ndim == 0 and sub_board.item() == Piece.EMPTY.value:
            return
        if np.all(sub_board == Piece.X.value):
            pygame.draw.line(screen, BoardRenderer.CROSS_COLOR, top_left, bottom_right, BoardRenderer.LINE_WIDTH // 3)
            pygame.draw.line(screen, BoardRenderer.CROSS_COLOR,
                             (top_left[0], bottom_right[1]),
                             (bottom_right[0], top_left[1]),
                             BoardRenderer.LINE_WIDTH // 3)

            return

        if np.all(sub_board == Piece.O.value):
            radius = min(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]) // 2 + padding // 2
            center = (top_left[0] + (bottom_right[0] - top_left[0]) // 2,
                      top_left[1] + (bottom_right[1] - top_left[1]) // 2)
            pygame.draw.circle(screen, BoardRenderer.CIRCLE_COLOR, center, radius, width=BoardRenderer.LINE_WIDTH // 3)
            assert radius > 0
            return

        third_delta_width = (bottom_right[0] - top_left[0]) // 3
        third_delta_height = (bottom_right[1] - top_left[1]) // 3
        for i in range(1, 3):
            # horizontal lines
            left = top_left[0], top_left[1] + i * third_delta_height
            right = bottom_right[0], top_left[1] + i * third_delta_height
            pygame.draw.line(screen, BoardRenderer.LINE_COLOR, left, right, line_width)

            # vertical lines
            top = top_left[0] + i * third_delta_width, top_left[1]
            bottom = top_left[0] + i * third_delta_width, bottom_right[1]
            pygame.draw.line(screen, BoardRenderer.LINE_COLOR, top, bottom, line_width)

        for i in range(3):
            for j in range(3):
                next_board_top_left = top_left[0] + j * third_delta_width, top_left[1] + i * third_delta_height
                next_board_bottom_right = (top_left[0] + third_delta_width + j * third_delta_width,
                                           top_left[1] + third_delta_height + i * third_delta_height)

                if next_board_bottom_right[0] - next_board_top_left[0] > padding and \
                    next_board_bottom_right[1] - next_board_top_left[1] > padding:
                    # pad if positions don't overlap
                    next_board_top_left = (next_board_top_left[0] + padding,
                                           next_board_top_left[1] + padding)
                    next_board_bottom_right = (next_board_bottom_right[0] - padding,
                                               next_board_bottom_right[1] - padding)

                BoardRenderer.draw_sub_board(screen, sub_board[i, j], next_board_top_left,
                                             next_board_bottom_right, line_width - 3, padding)

    def render(self, board: np.ndarray) -> None:
        """
        render the board to screen
        :param board: numpy array of shape (3, ..., 3) (even number of dimensions). Each entry should be if enum Piece
        """
        self.screen.fill(self.BG_COLOR)
        assert board.ndim % 2 == 0
        depth = board.ndim // 2
        self.draw_sub_board(self.screen, board, (0, 0), (self.height, self.width),
                            line_width=self.LINE_WIDTH, padding=self.PADDING // depth)
        pygame.display.update()
        self.clock.tick(self.render_fps)
        # TODO: change
        pygame.event.get()
        # run = True
        # while run:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             run = False

        # pygame.quit()


if __name__ == '__main__':
    board_renderer = BoardRenderer()
    depth = 2
    shape = [3, 3] * depth
    board = np.random.choice([Piece.X.value, Piece.O.value, Piece.EMPTY.value], size=shape)
    board[0, 0] = Piece.O.value
    board[0, 2] = Piece.X.value
    board_renderer.render(board)
