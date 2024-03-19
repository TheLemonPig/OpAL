from simulator import Simulator
from analysis import state_heatmap, success_metrics, plot_trends
from simulation_config import config_
import numpy as np
from itertools import product


def verbose_helper(simulator, results, n_reps, compare=None, **kwargs):
    if 'success_metrics' in kwargs:
        success_metrics(simulator, results, n_reps, compare=compare, **kwargs['success_metrics'])


def plot_helper(simulator, results, n_reps, compare=None, **kwargs):
    if 'state_heatmap' in kwargs:
        state_heatmap(simulator, results, n_reps, compare=compare, **kwargs['state_heatmap'])
    if 'trends' in kwargs:
        plot_trends(simulator, results, n_reps, compare=compare, **kwargs['trends'])


def grid_search(config, hyperparams):
    simulator = Simulator(config['environment_params'], config['model_params'])
    lists_of_hyperparams = {k: np.arange(*v) for k, v in hyperparams.items()}
    param_permutations = list(product(*lists_of_hyperparams.values()))
    n_combinations = len(param_permutations)
    assert n_combinations < 10e3, RuntimeError(f"Search space is impractically large: {n_combinations} combinations")
    scores = np.zeros((n_combinations, len(hyperparams)+1))
    for idx, params in enumerate(param_permutations):
        simulator.update_model_parameters({k: params[idx] for idx, k in enumerate(hyperparams.keys())})
        results = simulator.run(reps=config['n_reps'], steps=config['epochs'], seed=config['seed'])
        score = success_metrics(simulator, results, config['n_reps'], **config['verbose_params']['success_metrics'])
        for jdx, param in enumerate(params):
            scores[idx, jdx] = param
        scores[idx, -1] = score
    top_results = scores[np.argsort(scores[:, -1])[::-1]]
    head = min(10, n_combinations)
    leaderboard = [{k: top_results[idx, jdx] for jdx, k in enumerate(hyperparams.keys())} for idx in range(head)]
    print('\n--- Top Hyperparameter Settings ---\n')
    for i, result in enumerate(leaderboard):
        print(f'{i+1}) {[(k,np.round(v, decimals=5)) for k,v in result.items()]}\nAccuracy: {top_results[i,-1]}%')
    return scores


def compare(config, hyperparam: tuple):
    hyperparam_name, start, stop, step = hyperparam
    hyperparam_settings = np.arange(start, stop, step)
    simulator = Simulator(config['environment_params'], config['model_params'])
    compare_results = {}
    for hyperparam_value in hyperparam_settings:
        simulator.update_model_parameters({hyperparam_name: hyperparam_value})
        results = simulator.run(reps=config['n_reps'], steps=config['epochs'], seed=config['seed'])
        compare_results[(hyperparam_name, hyperparam_value)] = results
    env_name = config['environment_params'][0]['name']
    model_name = config['model_params'][0]['name']
    results = {env_name: {model_name:
                          [compare_results[(k, v)][env_name][model_name][0] for k, v in compare_results.keys()]
                          }}
    # TODO: Merge verbose_helper and plot_helper
    if config['verbose']:
        verbose_helper(simulator, results, config['n_reps'], compare=hyperparam, **config['verbose_params'])
    if config['plot']:
        plot_helper(simulator, results, config['n_reps'], compare=hyperparam, **config['plot_params'])


def run(config):
    simulator = Simulator(config['environment_params'], config['model_params'])
    results = simulator.run(reps=config['n_reps'], steps=config['epochs'], seed=config['seed'])
    if config['verbose']:
        verbose_helper(simulator, results, config['n_reps'], **config['verbose_params'])
    if config['plot']:
        plot_helper(simulator, results, config['n_reps'], **config['plot_params'])
    return results


if __name__ == "__main__":
    if config_['grid_search']:
        grid_search(config_, config_['grid_params'])
    elif config_['compare']:
        compare(config_, config_['compare_param'])
    else:
        run(config_)
