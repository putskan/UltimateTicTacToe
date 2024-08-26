from __future__ import annotations
import math
from typing import Any, Dict, Optional

import numpy as np
from pettingzoo import AECEnv

from agents.agent import Agent
from evaluate.example_usage import play
from utils.utils import get_action_mask, deepcopy_env

C = 2


class Node:
    """Node in the MCTS tree"""
    def __init__(self, env: AECEnv, action: int = None, parent: Node = None):
        """
        :param env: the environment of the game for this node
        :param action: the last action played in the env
        :param parent: the parent node of this node
        """
        self.value = 0
        self.n = 0
        self.state_env = env
        self.action = action
        self.children = []
        self.parent = parent

        _, _, termination, truncation, _ = env.last()
        self.is_terminal = termination or truncation

    def add_child(self, child: Node) -> None:
        """Add a child to the node"""
        self.children.append(child)

    def update(self, value):
        """Update the value and n count of the node"""
        self.value += value
        self.n += 1

    @property
    def ucb1(self):
        """Compute the UCB1 value of the node"""
        if self.n == 0:
            return float("inf")
        return self.value / self.n + C * math.sqrt(math.log(self.parent.n, math.e) / self.n)

    def get_max_ucb1_child(self):
        """Returns the child with the highest UCB1 value, or None if the node has no children
        (or if all of them are terminal nodes)"""
        if not self.children:
            return None, None

        max_i = 0
        max_ucb1 = float("-inf")

        for i, child in enumerate(self.children):
            ucb1 = child.ucb1

            if ucb1 > max_ucb1 and not child.is_terminal:
                max_ucb1 = ucb1
                max_i = i
        if max_ucb1 == float("-inf"):
            return None, None
        return self.children[max_i], max_i


class MCTSAgent(Agent):
    """Monte Carlo Tree Search agent"""
    def __init__(self, shuffle_move_order: bool = True,
                 n_iter_min=80, n_iter_max=400, action_n_thresh=12,
                 is_stochastic: bool = False,
                 *args, **kwargs):
        """
        :param shuffle_move_order: if True, adds the children (/actions) of each node in a random order
        :param n_iter: number of iterations to run the MCTS algorithm
        :param is_stochastic: if True, the agent chooses actions stochastically
        """
        super().__init__(*args, **kwargs)
        self.n_iter_min = n_iter_min
        self.n_iter_max = n_iter_max
        self.action_n_thresh = action_n_thresh
        self.shuffle_move_order = shuffle_move_order
        self.is_stochastic = is_stochastic

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int, curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:

        n_iter = self.n_iter_min
        if action_mask is not None:
            n_legal_actions = np.sum(action_mask)
            if n_legal_actions > self.action_n_thresh:
                n_iter = self.n_iter_max

        action = self.run_mcts(env, n_iter)
        return action

    def run_mcts(self, env: AECEnv, n_iter: int) -> float:
        """
        Run the MCTS algorithm
        :param env: playing environment
        :return: the next action to play
        """
        root_node = Node(env, None, None)
        self._expand(root_node)

        for _ in range(n_iter):
            leaf_node = self._select_expand(root_node)
            value = self._rollout(leaf_node)
            self._backpropagate(leaf_node, value)

        vals = [(child.value / child.n) if child.n else 0 for child in root_node.children]
        soft_vals = np.exp(vals) / sum(np.exp(vals))

        # choose action of the child with the highest value (deterministic)
        action = root_node.children[np.argmax(soft_vals)].action
        if self.is_stochastic:
            # choose action stochastically
            action = np.random.choice([child.action for child in root_node.children], p=soft_vals)
        return action

    def _expand(self,  node: Node) -> bool:
        """Expand the node by adding its children to the tree, if possible"""
        obs, reward, termination, truncation, info = node.state_env.last()
        # don't expand terminal nodes or previously expanded ones
        if node.is_terminal or node.children:
            return False

        action_mask = get_action_mask(obs, info)
        valid_actions = np.where(action_mask)[0]
        if self.shuffle_move_order:
            np.random.shuffle(valid_actions)

        for action in valid_actions.tolist():
            curr_env = deepcopy_env(node.state_env)
            curr_env.step(action)
            new_node = Node(curr_env, action, node)
            node.add_child(new_node)
        return True

    def _select_expand(self, root_node: Node) -> Node:
        """Select a node and expands it if possible, then returns the expanded node"""
        prev_node = root_node
        cur_node = root_node
        while cur_node is not None and cur_node.children:
            prev_node = cur_node
            cur_node, idx = cur_node.get_max_ucb1_child()

        if cur_node is not None:
            has_expanded = self._expand(cur_node)
            if has_expanded:
                cur_node2, idx2 = cur_node.get_max_ucb1_child()
                if cur_node2 is not None:
                    return cur_node2

        return cur_node or prev_node

    @staticmethod
    def _rollout(node: Node) -> float:
        """Simulate a game from the current node until the end and return the reward"""
        simulation_env = deepcopy_env(node.state_env)
        invert_reward = True  # invert the reward for the opponent

        while True:
            obs, reward, termination, truncation, info = simulation_env.last()
            if termination or truncation:
                if reward not in [0, 1, -1]:
                    raise ValueError(f"Unexpected reward {reward}")
                if invert_reward:
                    reward *= -1
                return (reward + 1) / 2  # 1 if win, 0 if lose, 0.5 if draw

            # simulate next move randomly
            action_mask = get_action_mask(obs, info)
            valid_actions = np.where(action_mask)[0]
            act = np.random.choice(valid_actions)
            simulation_env.step(act)
            invert_reward = not invert_reward

    @staticmethod
    def _backpropagate(leaf_node: Node, reward: float) -> None:
        """Update the value of the nodes from the leaf node to the root node"""
        curr_node = leaf_node

        while curr_node is not None:
            curr_node.update(reward)
            curr_node = curr_node.parent
            reward = 1 - reward


if __name__ == "__main__":
    from environments import ultimate_ttt
    from agents.choose_first_action_agent import ChooseFirstActionAgent
    from agents.random_agent import RandomAgent
    from agents.hierarchical_agent import HierarchicalAgent

    render_mode = "human"
    render_mode = None
    env = ultimate_ttt.env(render_mode=render_mode, depth=2)
    agents = [
        RandomAgent(),
        MCTSAgent(is_stochastic=True),
        # ChooseFirstActionAgent(),
        # HierarchicalAgent(),
    ]
    play(env, agents, n_games=5)
