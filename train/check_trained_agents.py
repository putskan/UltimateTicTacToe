from environments import ultimate_ttt
from evaluate.example_usage import play
from utils.utils import load_agent

if __name__ == "__main__":
    from agents.random_agent import RandomAgent
    from agents.reinforce_agent import ReinforceAgent
    from agents.choose_first_action_agent import ChooseFirstActionAgent

    file_path = f'trained_agents/ReinforceAgent_depth1_games1e5_hidden128/final.pickle'
    n_games = 200
    reinforce_agent: ReinforceAgent = load_agent(file_path)
    players = [RandomAgent(), reinforce_agent]
    # players = [reinforce_agent, ChooseFirstActionAgent()]
    env = ultimate_ttt.env(render_mode='human', depth=1)  # 'human', 'rgb_array', 'ansi', None

    print(f'Playing {n_games} games')
    print(f'Frist player: {players[0]}, Second player: {players[1]}')
    play(env, players, n_games=n_games)
    print()
    print(f'Playing {n_games} games')
    print(f'Frist player: {players[1]}, Second player: {players[0]}')
    play(env, players[::-1], n_games=n_games)
