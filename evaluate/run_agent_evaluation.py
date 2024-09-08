import os
from datetime import datetime

from agents.dqn_agent import DQNAgent
from environments import ultimate_ttt
from evaluate.evaluate_agents import AgentsEvaluator
from evaluation_functions.ae_winning_possibilities import AEWinningPossibilities
from evaluation_functions.dqn_evaluation import DQNEvaluation
from agents.choose_first_action_agent import ChooseFirstActionAgent
from agents.hierarchical_agent import HierarchicalAgent
from agents.random_agent import RandomAgent
from agents.reinforce_agent import ReinforceAgent
from agents.search_agents.alpha_beta import AlphaBeta
from agents.search_agents.mcts import MCTSAgent
from evaluation_functions.probabilistic_estimator import ProbabilisticEstimator
from evaluation_functions.sub_boards_won import SubBoardsWon
from evaluation_functions.win_loss_evaluation import WinLossEvaluation
from policy_functions.reinforce_policy import ReinforcePolicy
from policy_functions.soft_dqn_policy import SoftDQNPolicy
from utils.logger import get_logger
from utils.utils import load_agent

if __name__ == '__main__':
    """
    Run evaluation over the different agents
    """
    env = ultimate_ttt.env(render_mode=None, depth=2)
    reinforce_agent: ReinforceAgent = load_agent("../agents/trained_agents/reinforce/checkpoint_86000.pickle")
    dqn_agent: DQNAgent = load_agent("../agents/trained_agents/dqn/checkpoint_65000.pickle")

    ab_ae_agent = AlphaBeta(depth=2, evaluation_function=AEWinningPossibilities(),
                            shuffle_move_order=True,
                            agent_name='AB2-AE')

    ab_pe_agent = AlphaBeta(depth=2, evaluation_function=ProbabilisticEstimator(depth=3),
                            shuffle_move_order=True,
                            agent_name='AB2-PE-3')

    ab_sbw_agent = AlphaBeta(depth=2, evaluation_function=SubBoardsWon(),
                             shuffle_move_order=True,
                             agent_name='AB2-SBW')

    ab_wl_agent = AlphaBeta(depth=2, evaluation_function=WinLossEvaluation(),
                            shuffle_move_order=True,
                            agent_name='AB2-WL')

    ab_dqn_agent = AlphaBeta(depth=2, evaluation_function=DQNEvaluation(dqn_agent),
                             shuffle_move_order=True, agent_name='AB2-DQN')

    MCTS_sDQN_policy_DQN_eval = MCTSAgent(policy_function=SoftDQNPolicy(dqn_agent),
                                          evaluation_function=DQNEvaluation(dqn_agent),
                                          n_iter_min=40, n_iter_max=40,
                                          agent_name='Mcts-SoftDQN-DQN')

    MCTS_REINFORCE_policy_DQN_eval = MCTSAgent(policy_function=ReinforcePolicy(reinforce_agent),
                                               evaluation_function=DQNEvaluation(dqn_agent),
                                               n_iter_min=40, n_iter_max=40,
                                               agent_name='Mcts-Reinforce-DQN')

    MCTS_clean = MCTSAgent(policy_function=None,
                           evaluation_function=None,
                           n_iter_min=40, n_iter_max=40,
                           agent_name='Mcts-Clean')

    MCTS_PE_eval = MCTSAgent(policy_function=None,
                             evaluation_function=ProbabilisticEstimator(),
                             n_iter_min=40, n_iter_max=40,
                             agent_name='Mcts-PE-Eval')

    random_agent = RandomAgent()
    choose_first_action_agent = ChooseFirstActionAgent()
    hierarchical_agent = HierarchicalAgent()

    agents = [
        ab_ae_agent,
        ab_pe_agent,
        ab_sbw_agent,
        ab_wl_agent,
        ab_dqn_agent,
        MCTS_sDQN_policy_DQN_eval,
        MCTS_REINFORCE_policy_DQN_eval,
        MCTS_clean,
        MCTS_PE_eval,
        random_agent,
        choose_first_action_agent,
        hierarchical_agent,
    ]

    os.makedirs('logs', exist_ok=True)
    logger_file_name = f"logs/evaluate_agents_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger = get_logger("evaluate_agents", logger_file_name, log_to_console=True)
    agents_evaluator = AgentsEvaluator(agents, env, logger=logger, n_rounds=25)
    agents_evaluator.evaluate_agents()
