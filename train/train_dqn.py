import datetime

from pettingzoo.classic import tictactoe_v3

from agents.dqn_agent import DQNAgent
from train import train

if __name__ == '__main__':
    """
    script for training DQN
    """
    # env = ultimate_ttt.env(render_mode='human', depth=2, render_fps=10)
    env = tictactoe_v3.env(render_mode=None)
    renderable_env = tictactoe_v3.env(render_mode='human')
    env.reset()
    state_size = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    action_size = env.action_space(env.agents[0]).n
    discount_factor = 0.6
    date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    agent = DQNAgent(state_size=state_size, action_size=action_size, learning_rate=3e-4,
                     gamma=discount_factor, use_lr_scheduler=True)
    # noinspection PyCallingNonCallable
    train(env, agent, n_games=100_000, render_every=1_000,
          renderable_env=renderable_env, only_main_agent_to_replay=False,
          discount_factor=discount_factor,
          folder_to_save='logs',
          )
