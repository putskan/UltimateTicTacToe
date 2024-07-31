from typing import Tuple, Union

import numpy as np
from gymnasium import spaces

from pettingzoo import AECEnv
from environments.board import Board
from pettingzoo.utils import agent_selector, wrappers

from environments.board_renderer import BoardRenderer
from utils.piece import Piece


def env(render_mode: str = None, depth: int = 1, render_fps: int = 10) -> AECEnv:
    env = raw_env(render_mode=render_mode, depth=depth, render_fps=render_fps)
    if render_mode == 'ansi':
        env = wrappers.CaptureStdoutWrapper(env)

    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        'render_modes': ['human'],
        'name': 'ultimate_ttt',
        'is_parallelizable': False,
    }

    def __init__(self, render_mode: str | None = None,
                 depth: int = 1, render_fps: int = 10):
        super().__init__()
        self.render_fps = render_fps
        if render_mode is not None:
            self.board_renderer = BoardRenderer(render_fps=self.render_fps)
        self.depth = depth
        self.board = Board(depth=depth)
        self.forced_boards = None

        self.agents = ['player_1', 'player_2']
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        n_actions = np.prod(self.board.board_shape)
        self.action_spaces = {i: spaces.Discrete(n_actions)
                              for i in self.agents}
        # self.action_spaces = {i: spaces.Discrete(9) for i in self.agents}
        obs_shape = list(self.board.board_shape) + [2]
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    'observation': spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.int8),
                    'action_mask': spaces.Discrete(n_actions),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.render_mode = render_mode

    def observe(self, agent):
        board_vals = self.board.squares
        curr_player_idx = self.possible_agents.index(agent)
        if curr_player_idx == 0:
            curr_player_piece = Piece.X.value
            opponent_player_piece = Piece.O.value
        else:
            curr_player_piece = Piece.O.value
            opponent_player_piece = Piece.X.value

        curr_piece_board = board_vals == curr_player_piece
        opponent_piece_board = board_vals == opponent_player_piece

        observation = np.stack([curr_piece_board, opponent_piece_board], axis=self.depth + 1).astype(np.int8)

        legal_moves = self._legal_moves() if agent == self.agent_selection else np.array([])
        action_mask = legal_moves.flatten().astype(np.int8)
        return {'observation': observation, 'action_mask': action_mask}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self) -> np.ndarray:
        """
        filter boards we're not allowed to use + cells which are already occupied
        :return: boolean numpy array. True if legal, False otherwise.
        """
        if len(self.forced_boards) == 0:
            # relevant only for classic TTT
            assert self.depth == 1
            return self.board.squares == Piece.EMPTY.value

        legal_moves_mask = np.zeros_like(self.board.squares, dtype=np.bool)
        legal_moves_mask[*self.forced_boards] = True
        legal_moves_mask = np.logical_and(legal_moves_mask, self.board.squares == Piece.EMPTY.value)
        return legal_moves_mask

    def update_forced_boards(self, unraveled_action: Tuple[Union[int, np.signedinteger], ...]) -> None:
        """
        update the forced boards, so the next player must play a specified board
        """
        self.forced_boards = list(unraveled_action)[2:]
        for i in range(len(self.forced_boards), 0, -2):
            sub_board_indices = self.forced_boards[:i]
            if np.all(self.board.squares[*sub_board_indices] != Piece.EMPTY.value):
                # sub-game is finished
                self.forced_boards[i - 1] = slice(None)
                self.forced_boards[i - 2] = slice(None)
            else:
                break

    def step(self, action: int):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        unraveled_action = np.unravel_index(action, self.board.board_shape)
        self.board.play_turn(self.agents.index(self.agent_selection), unraveled_action)

        self.update_forced_boards(unraveled_action)

        next_agent = self._agent_selector.next()
        if self.board.check_game_over():
            winner = self.board.check_for_winner()
            if winner == 1:
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1
            elif winner == 2:
                self.rewards[self.agents[1]] += 1
                self.rewards[self.agents[0]] -= 1

            self.terminations = {i: True for i in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        if self.render_mode == 'human':
            self.render()

    def reset(self, seed=None, options=None):
        # reset environment
        if self.render_mode is not None:
            self.board_renderer = BoardRenderer(render_fps=self.render_fps)
        self.board = Board(self.depth)
        self.forced_boards = [slice(None)] * 2 * (self.depth - 1)
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    def close(self):
        pass

    def render(self):
        self.board_renderer.render(self.board.squares)


if __name__ == '__main__':
    import time
    environment = env(render_mode='human', depth=2, render_fps=5)
    environment.reset(seed=42)
    cumulative_rewards = [0, 0]
    curr_player_idx = 0

    for agent in environment.agent_iter():
        observation, reward, termination, truncation, info = environment.last()
        if termination or truncation:
            action = None
        else:
            action_mask = observation['action_mask']
            action = environment.action_space(agent).sample(mask=action_mask)

        cumulative_rewards[curr_player_idx] += reward
        environment.step(action)
        curr_player_idx = (curr_player_idx + 1) % 2

    environment.render()
    is_draw = cumulative_rewards[0] == cumulative_rewards[1]
    if is_draw:
        print('Draw!')
    else:
        winner = np.argmax(cumulative_rewards).item()
        print(f'Player {winner + 1} wins!')

    time.sleep(10)
    environment.close()
