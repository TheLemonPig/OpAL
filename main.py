from simulator import Simulator
from analysis import state_heatmap, success_metrics, plot_trends
from simulation_config import config_


def verbose_helper(simulator, results, n_reps, **kwargs):
    if 'success_metrics' in kwargs:
        success_metrics(simulator, results, n_reps, **kwargs['success_metrics'])


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
    if config['verbose']:
        verbose_helper(simulator, results, config['n_reps'], **config['verbose_params'])
    if config['plot']:
        plot_helper(simulator, results, config['n_reps'], **config['plot_params'])


if __name__ == "__main__":
    main(config_)
