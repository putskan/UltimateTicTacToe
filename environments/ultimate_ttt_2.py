from __future__ import annotations  # TODO?

import numpy as np
from gymnasium import spaces

from pettingzoo import AECEnv
from board import Board
from pettingzoo.utils import agent_selector, wrappers


def env(render_mode: str = None, depth: int = 1) -> AECEnv:
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode, depth=depth)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "ultimate_ttt",
        "is_parallelizable": False,
    }

    def __init__(self, render_mode: str | None = None,
                 depth: int = 1):
        super().__init__()
        self.depth = depth
        self.board = Board(depth=depth)

        self.agents = ["player_1", "player_2"]
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
                    "observation": spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.int8),
                    "action_mask": spaces.Discrete(n_actions),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        # TODO: legal moves vs action mask?
        # self.infos = {i: {"legal_moves": list(range(0, 9))} for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.render_mode = render_mode

    def observe(self, agent):
        board_vals = self.board.squares
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)  # TODO: change to enum, in board as well
        opp_p_board = np.equal(board_vals, opp_player + 1)

        observation = np.stack([cur_p_board, opp_p_board], axis=self.depth + 1).astype(np.int8)
        if agent != self.agent_selection:
            assert np.any(list(self.terminations.values())), list(self.terminations.values())


        # action_mask = legal_moves  # TODO: why int8 and not bool? remove?
        # action_mask = np.ravel_multi_index(np.nonzero(legal_moves), self.board.board_shape)
        legal_moves = self._legal_moves() if agent == self.agent_selection else np.array([])
        action_mask = legal_moves.flatten().astype(np.int8)

        return {"observation": observation, "action_mask": action_mask}

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
            return self.board.squares == 0

        legal_moves_mask = np.zeros_like(self.board.squares, dtype=np.bool)
        legal_moves_mask[self.forced_boards[self.agent_name_mapping[self.agent_selection]]] = True
        legal_moves_mask = np.logical_and(legal_moves_mask, self.board.squares == 0)
        return legal_moves_mask

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        # check if input action is a valid move (0 == empty spot)
        # assert self.board.squares[action] == 0, "played illegal move"
        print(action)
        action = np.unravel_index(action, self.board.board_shape)
        print(action)
        assert self._legal_moves()[action], "played illegal move"  # TODO: faster implementation?
        # play turn
        self.board.play_turn(self.agents.index(self.agent_selection), action)
        # self.forced_boards[self.agents[1 - self.agent_selection]] = None  # TODO: change

        # update infos
        # list of valid actions (indexes in board)
        # next_agent = self.agents[(self.agents.index(self.agent_selection) + 1) % len(self.agents)]
        next_agent = self._agent_selector.next()

        if self.board.check_game_over():
            winner = self.board.check_for_winner()
            if winner == 1:
                # agent 0 won
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1
            elif winner == 2:
                # agent 1 won
                self.rewards[self.agents[1]] += 1
                self.rewards[self.agents[0]] -= 1

            self.terminations = {i: True for i in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        # reset environment
        self.board = Board(self.depth)
        self.forced_boards = [slice(None)] * (self.depth - 1)
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
        print(self.board)


if __name__ == '__main__':
    environment = env(render_mode="human", depth=1)
    environment.reset(seed=42)
    for agent in environment.agent_iter():
        observation, reward, termination, truncation, info = environment.last()
        if termination or truncation:
            action = None
        else:
            action_mask = observation['action_mask']
            action = environment.action_space(agent).sample(mask=action_mask)

        environment.step(action)

    environment.close()
