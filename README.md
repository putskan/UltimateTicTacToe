# UltimateTTT
![Simulator](ultimate_ttt_simulation.gif)
## Installation
1. Make sure you have `Python 3.11` or above in your system.
2. Create a virtual environment with `Python 3.11` or above.
3. Install requirements specified in `requirements.txt`.

### Troubleshooting
* If clone fails using Windows, consider cloning the repository using `git clone -c core.protectNTFS=false <repo>`.
* If encountering `ModuleNotFoundError: No module named 'distutils'`, consult this [stackoverflow answer](https://stackoverflow.com/a/76691103/14984947) and re-install the requirements.

## Example usage
* In order to watch a game between two agents, run `evaluate/example_usage.py`.
* Feel free playing with the `depth` parameter to get a grasp of the N-depth-TTT version.

## Evaluating Agents
* In order to evaluate the different agents, and reproduce our plots, run `run_agent_evaluation.py`.
* Note this takes a while, so feel free to decrease the `n_rounds` parameter, alter the agent list etc.
