from agents.dqn_agent import DQNAgent
from agents.hierarchical_agent import HierarchicalAgent
from environments import ultimate_ttt
from evaluate.example_usage import play
from utils.utils import load_agent

if __name__ == "__main__":
    from agents.random_agent import RandomAgent
    from agents.reinforce_agent import ReinforceAgent
    from agents.choose_first_action_agent import ChooseFirstActionAgent

    file_path = f'logs/ReinforceAgent/2024-09-04_13-50-41/checkpoint_4000.pickle'
    # file_path = '/Users/benlahav/Desktop/uni/UltimateTicTacToe/train/logs/DQNAgent/2024-08-31_11-06-41/checkpoint_9.pickle'
    n_games = 200
    agent: ReinforceAgent = load_agent(file_path)
    agent.eval()
    oponents = [RandomAgent(), ChooseFirstActionAgent(), HierarchicalAgent()]
    for oponent in oponents:
        players = [agent, oponent]
        env = ultimate_ttt.env(render_mode=None, depth=2)  # 'human', 'rgb_array', 'ansi', None

        print(f'Playing {n_games} games')
        print(f'First player: {players[0]}, Second player: {players[1]}')
        play(env, players, n_games=n_games)
        print()
        print(f'Playing {n_games} games')
        print(f'First player: {players[1]}, Second player: {players[0]}')
        play(env, players[::-1], n_games=n_games)
        print()
