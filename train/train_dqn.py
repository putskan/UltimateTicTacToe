import datetime

from pettingzoo.classic import tictactoe_v3

from agents.dqn_agent import DQNAgent
from agents.probabilistic_estimator_dqn_agent import ProbabilisticEstimatorDQNAgent
from environments import ultimate_ttt
from models.dqn import PrevDQN
from train import train
from utils.utils import load_agent


def depth_2_training():
    env = ultimate_ttt.env(render_mode=None, depth=2)
    renderable_env = env

    env.reset()
    # state_size = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    action_size = env.action_space(env.agents[0]).n
    discount_factor = 0.6
    agent = ProbabilisticEstimatorDQNAgent(state_size=(3, 3, 2), action_size=action_size, learning_rate=3e-4,
                                           discount_factor=discount_factor, use_lr_scheduler=True, model_cls=PrevDQN)

    # previous_checkpoint: DQNAgent = load_agent('/Users/benlahav/Desktop/uni/UltimateTicTacToe/train/logs/DQNAgent/2024-08-31_12-54-55/checkpoint_65000.pickle')
    # agent.copy_networks(previous_checkpoint)

    # noinspection PyCallingNonCallable
    train(env, agent, n_games=100_000, render_every=1_000,
          renderable_env=renderable_env, only_main_agent_to_replay=False,
          discount_factor=discount_factor,
          folder_to_save='logs',
          save_checkpoint_every=5000,
          train_every=5,
          )


def depth_1_training():
    # env = tictactoe_v3.env(render_mode=None)
    # renderable_env = tictactoe_v3.env(render_mode='human')

    env = ultimate_ttt.env(render_mode=None, depth=1)
    renderable_env = ultimate_ttt.env(render_mode='human', depth=1, render_fps=2)
    env.reset()
    state_size = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    action_size = env.action_space(env.agents[0]).n
    discount_factor = 0.6
    agent = DQNAgent(state_size=state_size, action_size=action_size, learning_rate=3e-4,
                     discount_factor=discount_factor, use_lr_scheduler=True)
    # noinspection PyCallingNonCallable
    train(env, agent, n_games=100_000, render_every=1_000,
          renderable_env=renderable_env, only_main_agent_to_replay=False,
          discount_factor=discount_factor,
          folder_to_save='logs',
          save_checkpoint_every=5000,
          )


if __name__ == '__main__':
    """
    script for training DQN
    """
    depth_2_training()
