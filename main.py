from models import model_library
from environments import environment_library
from simulator import Simulator
from analysis import state_heatmap, success_metrics
from simulation_config import config_, params_


def setup(model_params, environment_params, simulator_config):
    simulators = []
    if simulator_config['type'] == 'solo':
        assert len(model_params) == 1 and len(environment_params) == 1
        model_config = model_params[0]
        environment_config = environment_params[0]
        model = model_library[model_config['name']](**model_config)
        environment = environment_library[environment_config['name']](**environment_config)
        simulator = Simulator(model, environment, **simulator_config)
        simulators.append(simulator)
    elif simulator_config['type'] == 'models':
        assert len(environment_params) == 1 and len(model_params) > 1
        raise NotImplementedError
    elif simulator_config['type'] == 'environments':
        assert len(environment_params) > 1 and len(model_params) == 1
        raise NotImplementedError
    else:
        raise KeyError("simulation type not available")
    return simulators


def run(simulators, epochs=100, verbose=True, plot=False):
    results = []
    for simulator in simulators:
        results.append(simulator.simulate(epochs))
    if verbose:
        verbose_helper(results, n_reps=1)
    if plot:
        plot_helper(results, n_reps=1)
    return results


def verbose_helper(results, n_reps, **kwargs):
    if 'success_metrics' in kwargs:
        success_metrics(results, n_reps, **kwargs['success_metrics'])


def plot_helper(results, n_reps, **kwargs):
    if 'state_heatmap' in kwargs:
        state_heatmap(results, n_reps, **kwargs['state_heatmap'])


def repeat(simulators, config, n_reps=10, epochs=100):
    # Results for each simulator stored separately
    simulator_results = [{'simulator': simulator} for simulator in simulators]
    for dic in simulator_results:
        dic['states'] = []
    # Results for each repetition stored inside dictionary for each simulator
    for n in range(n_reps):
        print(f"Repetition {n} out of {n_reps}")
        res = run(simulators, epochs, verbose=False, plot=False)
        for idx, dic in enumerate(simulator_results):
            dic['states'].append(res[idx]['states'])
    if config['verbose']:
        verbose_helper(simulator_results, n_reps, **config['verbose_params'])
    if config['plot']:
        plot_helper(simulator_results, n_reps, **config['plot_params'])


def grid_search():
    ...


def main(params, config):
    simulators = setup(*params)
    repeat(simulators, n_reps=config['n_reps'], epochs=config['epochs'], config=config)


if __name__ == "__main__":
    main(params_, config_)
