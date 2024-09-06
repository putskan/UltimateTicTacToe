from __future__ import annotations

import math
from typing import Any, Dict, Optional, List, Union

import numpy as np
import torch
from pettingzoo import AECEnv

from agents.agent import Agent
from agents.dqn_agent import DQNAgent
from agents.reinforce_agent import ReinforceAgent
from evaluation_functions.dqn_evaluation import DQNEvaluation
from evaluation_functions.evaluation_function import EvaluationFunction
from policy_functions.policy_function import PolicyFunction
from policy_functions.reinforce_policy import ReinforcePolicy
from policy_functions.soft_dqn_policy import SoftDQNPolicy
from utils.constants import INF
from utils.utils import get_action_mask, deepcopy_env, load_agent


class Node:
    """Node in the MCTS tree"""
    def __init__(self, state_env: AECEnv, policy_vals: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 action: int = None, parent: Node = None, c: float = 2, shuffle_move_order: bool = False):
        """
        :param state_env: the environment of the game for this node. Note: This object should implement an eqal
                    method and hash method. For uttt environment, the env should be unwrapped.
        :param action: the last action played in the env
        :param parent: the parent node of this node
        :param c: the exploration parameter
        :param shuffle_move_order: whether to shuffle the order of valid actions
        """
        self.value = 0
        self.n = 0
        self.state_env = state_env
        self.action: int = action
        self.parent = parent
        self.c = c
        self.policy_vals = policy_vals.tolist()
        self.depth = self.parent.depth + 1 if self.parent is not None else 1

        obs, _, termination, truncation, info = state_env.last()
        self.is_terminal = termination or truncation
        self.valid_actions = np.where(get_action_mask(obs, info))[0]
        if shuffle_move_order:
            np.random.shuffle(self.valid_actions)
        self.valid_actions = self.valid_actions.tolist()
        self.children: List[Optional[Node]] = [None] * len(self.valid_actions)
        self.action_index_mapping = {action: i for i, action in enumerate(self.valid_actions)}

    def get_child_by_action(self, action: int) -> Node:
        """Get the child node by the action"""
        index = self.action_index_mapping[action]
        return self.children[index]

    def add_child(self, child: Node) -> None:
        """Add a child to the node"""
        if self.get_child_by_action(child.action) is None:
            self.children[self.action_index_mapping[child.action]] = child

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
        for action in self.valid_actions:
            child = self.get_child_by_action(action)
            if self.policy_vals is None:
                val = float("inf") if child is None else child.ucb1
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

    TERMINATING_REWARDS = (0, 1, -1)

    def __init__(self, shuffle_move_order: bool = False,
                 n_iter_min=80, n_iter_max=400, action_n_thresh=12,
                 is_stochastic: bool = False, c: float = 2,
                 epsilon: float = 0.5, min_epsilon: float = 0.1,
                 evaluation_function: EvaluationFunction = None,
                 policy_function: PolicyFunction = None,
                 *args, **kwargs):
        """
        :note: If you provide an evaluation function, make sure its return values are centered around 0.
               Also, it is best if the values are in the range [-1, 1] (meaning -1 for loss and 1 for win),
               and if not, make sure to adjust the c parameter that balances exploration vs. exploitation.
        :param shuffle_move_order: if True, shuffles the valid actions order of each node.
               Should be True if policy function is not provided.
        :param n_iter: number of iterations to run the MCTS algorithm
        :param is_stochastic: if True, the agent chooses actions stochastically
        :param c: the exploration parameter
        :param epsilon: controls the amount of exploration for higher nodes in the tree
        :param min_epsilon: the minimum value of epsilon for exploration
        :param evaluation_function: (optional) the evaluation function to use in the MCTS algorithm.
                                    If not provided, uses random rollout.
        :param policy_function: (optional) the policy function to use in the MCTS algorithm.
        """
        super().__init__(*args, **kwargs)
        if 'agent_name' not in kwargs:
            details = f'Policy {policy_function.__class__.__name__}' if policy_function is not None else ''
            details += f' Eval {evaluation_function.__class__.__name__}' if evaluation_function is not None else ''
            self.agent_name = f'{self.__class__.__name__} min {n_iter_min} max {n_iter_max} ({details})'
        self.n_iter_min = n_iter_min
        self.n_iter_max = n_iter_max
        self.action_n_thresh = action_n_thresh
        self.shuffle_move_order = shuffle_move_order
        self.is_stochastic = is_stochastic
        self.eval_fn = evaluation_function
        self.policy_fn = policy_function
        self.c = c
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon

        self.env_transposition_table = dict()
        self.curr_root_node = None

    def _update_root_node(self, child_env: AECEnv) -> None:
        """Update the root node to the child node with the given environment"""
        for action in self.curr_root_node.valid_actions:
            child = self.curr_root_node.get_child_by_action(action)
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
        root_node = Node(copy_env, policy_values, None, None, c=self.c, shuffle_move_order=self.shuffle_move_order)
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
        for action in self.curr_root_node.valid_actions:
            child = self.curr_root_node.get_child_by_action(action)
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
        if node.is_terminal or node.get_child_by_action(action_child_to_expand) is not None:
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

        new_node = Node(curr_env, policy_values, action_child_to_expand, node, c=self.c, shuffle_move_order=self.shuffle_move_order)
        node.add_child(new_node)
        return True

    def _select_expand(self, root_node: Node) -> Node:
        """Select a node and expands it if possible, then returns the expanded node"""
        prev_node = root_node
        cur_node = root_node
        action_to_expand = None
        while cur_node is not None and not cur_node.is_terminal:
            prev_node = cur_node
            action_to_expand = self._select_next_action(cur_node)
            cur_node = cur_node.get_child_by_action(action_to_expand)
        assert action_to_expand is not None, "No action to expand!"
        self._expand(prev_node, action_to_expand)
        return prev_node.get_child_by_action(action_to_expand)

    def _select_next_action(self, node: Node) -> int:
        """Select the next action based on the current node"""
        d_epsilon = self.epsilon / node.depth
        if d_epsilon > self.min_epsilon and d_epsilon > np.random.rand():
                return np.random.choice(node.valid_actions)
        return node.get_max_action()

    def _rollout(self, node: Node) -> float:
        """If an evaluation function is provided, call it on the node's env. Otherwise, simulate the game
        from the current node until the end and return the reward"""
        if self.eval_fn is not None:
            observation, reward, termination, truncation, info = node.state_env.last()
            if termination or truncation:
                val = node.state_env.rewards[node.state_env.agent_selection]
                return (val + 1) / 2
            curr_agent_idx = node.state_env.agent_selection
            # TODO: check that observation and agent index are correct
            val = self.eval_fn(node.state_env, observation, curr_agent_idx)
            return (val + 1) / 2

        simulation_env = deepcopy_env(node.state_env)
        invert_reward = True  # invert the reward for the opponent

        while True:
            obs, reward, termination, truncation, info = simulation_env.last()
            if termination or truncation:
                if reward not in self.TERMINATING_REWARDS:
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
    import datetime
    from agents.search_agents.alpha_beta import AlphaBeta
    from evaluate.evaluate_agents import AgentsEvaluator
    from evaluation_functions.ae_winning_possibilities import AEWinningPossibilities
    from evaluation_functions.constant_evaluation import ConstantEvaluation
    from evaluation_functions.probabilistic_estimator import ProbabilisticEstimator
    from evaluation_functions.sub_boards_won import SubBoardsWon
    from utils.logger import get_logger
    from environments import ultimate_ttt
    from agents.choose_first_action_agent import ChooseFirstActionAgent
    from agents.random_agent import RandomAgent
    from agents.hierarchical_agent import HierarchicalAgent

    render_mode = None
    env = ultimate_ttt.env(render_mode=render_mode, depth=2)
    dqn_agent: DQNAgent = load_agent("../../agents/trained_agents/dqn/checkpoint_65000.pickle")
    reinforce_agent: ReinforceAgent = load_agent("../../agents/trained_agents/reinforce/checkpoint_86000.pickle")
    agents = [
        MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
                  n_iter_min=40, n_iter_max=40, agent_name='mcts'),
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_.1', c=.1),
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_.5', c=.5),
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_1', c=1),
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_sqrt(2)', c=math.sqrt(2)),
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_2', c=2),
        #
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_3', c=3),
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_5', c=5),
        # MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
        #           n_iter_min=40, n_iter_max=40, agent_name='mcts_10', c=10),

        MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
                  evaluation_function=DQNEvaluation(dqn_agent),
                  n_iter_min=40, n_iter_max=40, agent_name='mcts_dqn'),

    ]
    # logger_file_name = f"../../evaluate/logs/evaluate_agents_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # logger = get_logger("evaluate_agents", logger_file_name, log_to_console=True)
    # agents_evaluator = AgentsEvaluator(agents, env, logger=logger, n_rounds=10)
    # agents_evaluator.evaluate_agents()
    agents_evaluator = AgentsEvaluator(agents, env, n_rounds=1)
    agents_evaluator.evaluate_agents()
