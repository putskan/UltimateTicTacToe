from __future__ import annotations
import math
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from pettingzoo import AECEnv

from agents.agent import Agent
from agents.reinforce_agent import ReinforceAgent
from agents.search_agents.alpha_beta import AlphaBeta
from evaluate.example_usage import play
from evaluation_functions.ae_winning_possibilities import AEWinningPossibilities
from evaluation_functions.evaluation_function import EvaluationFunction
from evaluation_functions.probabilistic_estimator import ProbabilisticEstimator
from evaluation_functions.sub_boards_won import SubBoardsWon
from policy_functions.policy_function import PolicyFunction
from utils.constants import INF
from utils.utils import get_action_mask, deepcopy_env, load_agent


class Node:
    """Node in the MCTS tree"""
    def __init__(self, state_env: AECEnv, policy_vals: Optional[np.ndarray] = None,
                 action: int = None, parent: Node = None, c: float = 2):
        """
        :param state_env: the environment of the game for this node. Note: This object should implement an eqal
                    method and hash method. For uttt environment, the env should be unwrapped.
        :param action: the last action played in the env
        :param parent: the parent node of this node
        :param c: the exploration parameter
        """
        self.value = 0
        self.n = 0
        self.state_env = state_env
        self.action: int = action
        self.parent = parent
        self.c = c
        self.policy_vals = policy_vals

        obs, _, termination, truncation, info = state_env.last()
        action_mask = get_action_mask(obs, info)
        self.is_terminal = termination or truncation
        self.action_mask = action_mask
        self.children: List[Optional[Node]] = [None] * len(action_mask)

    def add_child(self, child: Node) -> None:
        """Add a child to the node"""
        if self.children[child.action] is None:
            self.children[child.action] = child

    def update(self, value):
        """Update the value and n count of the node"""
        self.value += value
        self.n += 1

    def uct(self, prob: float):
        """Compute the UCT value of the node"""
        avg_value = self.value / self.n if self.n > 0 else 0
        return avg_value + (self.c * prob) / (1 + self.n)

    @property
    def ucb1(self):
        """Compute the UCB1 value of the node"""
        if self.n == 0:
            return float("inf")
        return self.value / self.n + self.c * math.sqrt(math.log(self.parent.n, math.e) / self.n)

    def get_max_action(self) -> int:
        """Returns the child with the highest UCB1 value, or None if the node has no children
        (or if all of them are terminal nodes)"""
        max_val = float("-inf")
        max_action = None
        for action, child in enumerate(self.children):
            if self.policy_vals is None:
                if self.action_mask[action] == 0:
                    val = float("-inf")  # illegal moves are the worst
                elif child is None:
                    val = float("inf")  # unexplored nodes are with high priority
                else:
                    val = child.ucb1
            else:  # value is according to policy or UCT value
                prob = self.policy_vals[action]
                val = prob if child is None else child.uct(prob)
            if val > max_val:
                max_val = val
                max_action = action
        assert max_action is not None, "No action found with max UCB1/UCT value"
        return max_action

    def __eq__(self, other: Node) -> bool:
        return self.state_env == other.state_env

    def __hash__(self) -> int:
        return hash(self.state_env)


class MCTSAgent(Agent):
    """Monte Carlo Tree Search agent"""
    def __init__(self, shuffle_move_order: bool = True,
                 n_iter_min=80, n_iter_max=400, action_n_thresh=12,
                 is_stochastic: bool = False, c: float = 2,
                 evaluation_function: EvaluationFunction = None,
                 policy_function: PolicyFunction = None,
                 *args, **kwargs):
        """
        :note: If you provide an evaluation function, make sure its return values are centered around 0.
               Also, it is best if the values are in the range [0, 1] (meaning 0 for loss and 1 for win),
               and if not, make sure to adjust the c parameter that balances exploration vs. exploitation.
        :param shuffle_move_order: if True, adds the children (/actions) of each node in a random order
        :param n_iter: number of iterations to run the MCTS algorithm
        :param is_stochastic: if True, the agent chooses actions stochastically
        :param c: the exploration parameter
        :param evaluation_function: (optional) the evaluation function to use in the MCTS algorithm.
                                    If not provided, uses random rollout.
        :param policy_function: (optional) the policy function to use in the MCTS algorithm.
        """
        super().__init__(*args, **kwargs)
        self.n_iter_min = n_iter_min
        self.n_iter_max = n_iter_max
        self.action_n_thresh = action_n_thresh
        self.shuffle_move_order = shuffle_move_order
        self.is_stochastic = is_stochastic
        self.eval_fn = evaluation_function
        self.policy_fn = policy_function
        self.c = c

        self.env_transposition_table = dict()
        self.curr_root_node = None

    def _update_root_node(self, child_env: AECEnv) -> None:
        """Update the root node to the child node with the given environment"""
        for child in self.curr_root_node.children:
            if child is not None and child.state_env == child_env:
                self.curr_root_node = child
                self.curr_root_node.parent = None
                return

        # print("Child env not found in the root node's children!")
        self.curr_root_node = None

    def _add_env_to_transposition_table(self, unwrapped_env: AECEnv) -> AECEnv:
        """
        Add the environment to the transposition table if it is not already there.
        :param unwrapped_env: the environment to add. Note: This env should be unwrapped, i.e. as in the Node class.
        """
        if unwrapped_env not in self.env_transposition_table:
            self.env_transposition_table[unwrapped_env] = unwrapped_env
        return self.env_transposition_table[unwrapped_env]

    def reset(self) -> None:
        """Reset the agent to its initial state"""
        super().reset()
        self.env_transposition_table = dict()
        self.curr_root_node = None

    def play(self, env: AECEnv, obs: Any, curr_agent_idx: int, curr_agent_str: str, action_mask: Optional[np.ndarray],
             info: Dict[str, Any]) -> Any:
        if self.curr_root_node is not None:
            self._update_root_node(env.unwrapped)

        n_iter = self.n_iter_min
        if action_mask is not None:
            n_legal_actions = np.sum(action_mask)
            if n_legal_actions > self.action_n_thresh:
                n_iter = self.n_iter_max

        action = self.run_mcts(env, n_iter)
        return action

    def _create_root_node(self, env: AECEnv) -> Node:
        """Create the root node of the MCTS tree"""
        copy_env = deepcopy_env(env)
        obs, reward, termination, truncation, info = copy_env.last()
        action_mask = get_action_mask(obs, info)
        if isinstance(obs, dict):
            obs = obs["observation"]
        policy_values = self.policy_fn(obs, action_mask) if self.policy_fn is not None else None
        copy_env = self._add_env_to_transposition_table(copy_env.unwrapped)
        root_node = Node(copy_env, policy_values, None, None, c=self.c)
        return root_node

    def _calc_root_children_soft_vals(self) -> np.ndarray:
        """Calculate the softmax values of the children of the given node"""
        vals = np.array([child.value / child.n if child is not None and child.n and not child.is_terminal
                         else -INF for child in self.curr_root_node.children])
        soft_vals = np.exp(vals) / np.exp(vals).sum()
        return soft_vals

    def run_mcts(self, original_env: AECEnv, n_iter: int) -> float:
        """
        Run the MCTS algorithm
        :param original_env: playing environment
        :return: the next action to play
        """
        if self.curr_root_node is None:
            self.curr_root_node = self._create_root_node(original_env)
            max_action = self.curr_root_node.get_max_action()
            self._expand(self.curr_root_node, max_action)
        root_node = self.curr_root_node

        for _ in range(n_iter):
            leaf_node = self._select_expand(root_node)
            value = self._rollout(leaf_node)
            self._backpropagate(leaf_node, value)

        immediate_action = self._check_for_immediate_winning_action()
        if immediate_action is not None:
            return immediate_action

        soft_vals = self._calc_root_children_soft_vals()
        # choose action of the child with the highest value (deterministic)
        action_i = np.argmax(soft_vals)
        if self.is_stochastic:
            # choose action stochastically
            action_i = np.random.choice(len(root_node.children), p=soft_vals)
        chosen_child = root_node.children[action_i]
        assert chosen_child is not None, "Chosen child should not be None!"
        if not chosen_child.is_terminal:
            self._update_root_node(chosen_child.state_env)
        return chosen_child.action

    def _check_for_immediate_winning_action(self) -> Optional[int]:
        best_action = None
        for child in self.curr_root_node.children:
            if child is not None and child.is_terminal:
                _, reward, _, _, _ = child.state_env.last()
                reward *= -1
                assert reward != -1, "Loss should not happen for children of the root node!"
                if reward == 1:  # win
                    return child.action
                else:  # draw
                    best_action = child.action
        return best_action

    def _expand(self,  node: Node, action_child_to_expand: int) -> bool:
        """Expand the node by adding its children to the tree, if possible"""
        # don't expand terminal nodes or previously expanded ones
        if node.is_terminal or node.children[action_child_to_expand] is not None:
            return False

        curr_env = deepcopy_env(node.state_env)
        curr_env.step(action_child_to_expand)
        obs, reward, termination, truncation, info = curr_env.last()
        action_mask = get_action_mask(obs, info)
        if isinstance(obs, dict):
            obs = obs["observation"]
        policy_values = self.policy_fn(obs, action_mask) if self.policy_fn is not None else None
        # add env to transposition table
        curr_env = self._add_env_to_transposition_table(curr_env)
        assert curr_env in self.env_transposition_table, "Node not in the transpo table!"

        new_node = Node(curr_env, policy_values, action_child_to_expand, node, c=self.c)
        node.add_child(new_node)
        return True

    def _select_expand(self, root_node: Node) -> Node:
        """Select a node and expands it if possible, then returns the expanded node"""
        prev_node = root_node
        cur_node = root_node
        action_to_expand = None
        while cur_node is not None and not cur_node.is_terminal:
            prev_node = cur_node
            action_to_expand = cur_node.get_max_action()
            if cur_node.children[action_to_expand] is not None:
                cur_node = cur_node.children[action_to_expand]
            else:
                cur_node = None
        assert action_to_expand is not None, "No action to expand!"
        has_expanded = self._expand(prev_node, action_to_expand)
        return prev_node.children[action_to_expand] if has_expanded else prev_node

    def _rollout(self, node: Node) -> float:
        """If an evaluation function is provided, call it on the node's env. Otherwise, simulate the game
        from the current node until the end and return the reward"""
        if self.eval_fn is not None:
            observation, reward, termination, truncation, info = node.state_env.last()
            curr_agent_idx = node.state_env.agent_selection
            # TODO: check that observation and agent index are correct
            val = self.eval_fn(node.state_env, observation, curr_agent_idx)
            return val

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
    class ReinforcePolicy(PolicyFunction):
        def __init__(self):
            super().__init__("REINFORCE")
            self.reinforce_agent: ReinforceAgent = load_agent("../../train/logs/ReinforceAgent/2024-09-04_12-56-44/checkpoint_15000.pickle")
            # self.reinforce_agent: ReinforceAgent = load_agent("../../train/logs/ReinforceAgent/2024-09-04_13-50-41/checkpoint_1500.pickle")

        def __call__(self, state: Any, action_mask: Any, *args, **kwargs) -> np.ndarray:
            state = torch.tensor(state, dtype=torch.float32, device=self.reinforce_agent.device).unsqueeze(0)
            action_mask = torch.BoolTensor(action_mask, device=self.reinforce_agent.device).unsqueeze(0)
            return self.reinforce_agent.apply_policy_net(state, action_mask).flatten().detach()

    render_mode = "human"
    render_mode = None
    env = ultimate_ttt.env(render_mode=render_mode, depth=1)
    agents = [
        # RandomAgent(),
        MCTSAgent(policy_function=ReinforcePolicy(), n_iter_min=15, is_stochastic=True),
        ChooseFirstActionAgent(),
        # HierarchicalAgent(),
        # MCTSAgent(),
        # MCTSAgent(policy_function=ReinforcePolicy(), n_iter_min=20, n_iter_max=20),
        # AlphaBeta(depth=1, evaluation_function=ProbabilisticEstimator()),
        # MCTSAgent(n_iter_min=20, n_iter_max=160),
    ]
    play(env, agents, n_games=20)
