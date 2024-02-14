from simulator import Simulator
from analysis import state_heatmap, success_metrics
from simulation_config import config_


def setup(environment_params, model_params):
    #models = []
    #environments = []
    # for e_params in environment_params:
    #     environments.append(environment_library[e_params['name']](**e_params))
    # for m_params in model_params:
    #     models.append(model_library[m_params['name']](**m_params))
    simulator = Simulator(environment_params, model_params)
    return simulator


def verbose_helper(simulator, results, n_reps, **kwargs):
    if 'success_metrics' in kwargs:
        success_metrics(simulator, results, n_reps, **kwargs['success_metrics'])


def plot_helper(simulator, results, n_reps, **kwargs):
    if 'state_heatmap' in kwargs:
        state_heatmap(simulator, results, n_reps, **kwargs['state_heatmap'])


def grid_search():
    ...


def main(config):
    simulator = setup(config['environment_params'], config['model_params'])
    results = simulator.run(reps=config['n_reps'], steps=config['epochs'])
    if config['verbose']:
        verbose_helper(simulator, results, config['n_reps'], **config['verbose_params'])
    if config['plot']:
        plot_helper(simulator, results, config['n_reps'], **config['plot_params'])


if __name__ == "__main__":
    main(config_)
