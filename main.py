from simulator import Simulator
from analysis import state_heatmap, action_heatmap, weight_heatmap, policy_heatmap, success_metrics, plot_trends
from simulation_config import config_
import numpy as np
from itertools import product


def verbose_helper(config, results, n_reps, **kwargs):
    if 'success_metrics' in kwargs:
        return success_metrics(config, results, n_reps, **kwargs['success_metrics'])


def plot_helper(config, results, n_reps, **kwargs):
    if 'state_heatmap' in kwargs:
        state_heatmap(config, results, n_reps, **kwargs['state_heatmap'])
    if 'action_heatmap' in kwargs:
        action_heatmap(config, results, n_reps, **kwargs['action_heatmap'])
    if 'weight_heatmap' in kwargs:
        weight_heatmap(config, results, n_reps, **kwargs['weight_heatmap'])
    if 'policy_heatmap' in kwargs:
        policy_heatmap(config, results, n_reps, **kwargs['policy_heatmap'])
    if 'trends' in kwargs:
        plot_trends(config, results, n_reps, **kwargs['trends'])


def grid_search():
    ...


def main(config):
    simulator = Simulator(config['environment_params'], config['model_params'])
    results = simulator.run(reps=config['n_reps'], steps=config['epochs'], seed=config['seed'], thin=config['thin'])
    res = None
    if config['verbose']:
        res = verbose_helper(config, results, config['n_reps'], **config['verbose_params'])
    if config['plot']:
        plot_helper(config, results, config['n_reps'], **config['plot_params'])
    return res


if __name__ == "__main__":
    main(config_)
