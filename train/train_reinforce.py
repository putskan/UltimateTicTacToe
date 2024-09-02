import datetime
import numpy as np

from agents.reinforce_agent import ReinforceAgent
from environments import ultimate_ttt
from train import train

if __name__ == '__main__':
    """
    script for training REINFORCE
    """
    env = ultimate_ttt.env(render_mode=None, depth=1)
    renderable_env = ultimate_ttt.env(render_mode='human', depth=1, render_fps=2)

    env.reset()
    state_shape = env.unwrapped.observation_spaces[env.agents[0]].spaces['observation'].shape
    action_size = env.action_space(env.agents[0]).n
    state_size = np.prod(state_shape)
    hidden_size = 64
    batch_size = 128
    discount_factor = 0.6
    learning_rate = 3e-4
    agent = ReinforceAgent(state_size=state_size, action_size=action_size,
                           hidden_size=hidden_size, batch_size=batch_size,
                           learning_rate=learning_rate, discount_factor=discount_factor,
                           use_lr_scheduler=True)
    # noinspection PyCallingNonCallable
    train(env, agent, n_games=100_000, render_every=1_000,
          renderable_env=renderable_env, only_main_agent_to_replay=False,
          discount_factor=discount_factor,
          folder_to_save='logs',
          save_checkpoint_every=5000,
          )
