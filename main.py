from simulator import Simulator
from analysis import state_heatmap, success_metrics, plot_trends
from simulation_config import config_

import numpy as np
from itertools import product


def verbose_helper(simulator, results, n_reps, **kwargs):
    if 'success_metrics' in kwargs:
        return success_metrics(simulator, results, n_reps, **kwargs['success_metrics'])


def plot_helper(simulator, results, n_reps, **kwargs):
    if 'state_heatmap' in kwargs:
        state_heatmap(simulator, results, n_reps, **kwargs['state_heatmap'])
    if 'trends' in kwargs:
        plot_trends(simulator, results, n_reps, **kwargs['trends'])



def grid_search():
    ...


def main(config):
    simulator = Simulator(config['environment_params'], config['model_params'])
    results = simulator.run(reps=config['n_reps'], steps=config['epochs'], seed=config['seed'])
    res = None
    if config['verbose']:
        res = verbose_helper(simulator, results, config['n_reps'], **config['verbose_params'])
    if config['plot']:
        plot_helper(simulator, results, config['n_reps'], **config['plot_params'])
    return res


if __name__ == "__main__":
    main(config_)
    # # env_params = [{(2, 3): 0.8, (0, 3): 0.7}, {(2, 3): 0.3, (0, 3): 0.2}]
    # env_params = [{(0, 5): 0.2, (2, 7): 0.1, (4, 5): 0.3},
    #               {(0, 5): 0.7, (2, 7): 0.6, (4, 5): 0.8}]
    # hyperparams = {'alpha': (0.4, 0.7, 0.1),  # 'alpha_c': (0.5, 0.7, 0.1),
    #                'beta': (4.0, 7.0, 1.0), 'gamma': (0.7, 1.0, 0.1)}
    # # hyperparams = {'k': (5.0, 9.0, 4.0), 'phi': (1.4, 1.9, 0.1)}
    # lists_of_hyperparams = {k: np.arange(*v) for k, v in hyperparams.items()}
    # param_permutations = list(product(*lists_of_hyperparams.values()))
    # model = config_['model_params'][0]
    # environment = config_['environment_params'][0]
    # meta_results = []
    # for env_par in env_params:
    #     environment['terminal_states'] = env_par
    #     config_['environment_params'][0] = environment
    #     results = np.zeros((len(param_permutations),))
    #     for idx, params in enumerate(param_permutations):
    #         if idx % 10 == 0:
    #             print(f'\n{idx} out of {len(param_permutations)}\n')
    #         new_params = {k: np.round(params[idx], 5) for idx, k in enumerate(hyperparams.keys())}
    #         for param in new_params.keys():
    #             if param == 'alpha' and model['model'].startswith("OpAL"):
    #                 model[param+'_g'] = new_params[param]
    #                 model[param + '_n'] = new_params[param]
    #                 model[param + '_c'] = new_params[param]
    #             else:
    #                 model[param] = new_params[param]
    #         config_['model_params'][0] = model
    #
    #         print(new_params)
    #         success_rate = main(config_)
    #         results[idx] = success_rate
    #     meta_results.append(results)
    # collective = meta_results[0] + meta_results[1]
    # args = np.argsort(collective)
    # print(f'\n{model["name"]}:\n')
    # for i in range(20):
    #     arg = args[-(i+1)]
    #     print(f'{np.round(param_permutations[arg], 5)}: {meta_results[0][arg]}, {meta_results[1][arg]}, {collective[arg]}')
