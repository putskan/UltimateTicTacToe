import unittest

from environments import ultimate_ttt


class TestEnv(unittest.TestCase):

    def test_eq(self):
        env = ultimate_ttt.env(render_mode=None, depth=2, render_fps=10)
        unwrapped_env = env.unwrapped
        unwrapped_env_2 = ultimate_ttt.env(depth=2, render_fps=6).unwrapped
        unwrapped_env_3 = ultimate_ttt.env(depth=3, render_fps=5).unwrapped

        env.reset()
        unwrapped_env_2.reset()
        unwrapped_env_3.reset()

        assert unwrapped_env == unwrapped_env
        assert unwrapped_env_2 == unwrapped_env
        assert unwrapped_env_2 != unwrapped_env_3

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action_mask = observation['action_mask']
                action = env.action_space(agent).sample(mask=action_mask)

            env.step(action)
            assert unwrapped_env_2 != unwrapped_env

        assert unwrapped_env_2 != unwrapped_env
        env.reset()
        assert unwrapped_env_2 == unwrapped_env
        env.close()


if __name__ == '__main__':
    unittest.main()
