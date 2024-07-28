import time

from agents.choose_first_action_agent import ChooseFirstActionAgent
from agents.random_agent import RandomAgent
from pettingzoo.classic import tictactoe_v3

from utils.utils import get_action_mask

if __name__ == '__main__':
    seed = 42
    env = tictactoe_v3.env(render_mode='human')  # 'human', 'rgb_array', 'ansi'
    env.reset(seed=seed)

    players = [RandomAgent(), ChooseFirstActionAgent()]
    curr_player_idx = 0
    obs = env.reset()

    for curr_agent_str in env.agent_iter():
        curr_player = players[curr_player_idx]
        observation, reward, termination, truncation, info = env.last()
        action_mask = get_action_mask(observation, info)
        done = termination or truncation
        if done:
            action = None
        else:
            action = curr_player.play(env, obs, curr_agent_str, action_mask)

        env.step(action)
        curr_player_idx = (curr_player_idx + 1) % len(players)

        # visualize
        env.render()
        time.sleep(0.0001)

    env.close()
