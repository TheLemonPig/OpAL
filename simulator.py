from tqdm import tqdm
from copy import deepcopy
import numpy as np

from models import model_library
from environments import environment_library
from typing import Dict


class Simulator:

    def __init__(self, environments, models):
        self.environments = environments
        self.models = models
        self.results = {env['name']:
                        {model['name']: [] for model in models}
                        for env in environments}

    def run(self, reps, steps, seed=None):
        self.results = {env['name']:
                        {model['name']: [] for model in self.models}
                        for env in self.environments}
        for env_params in self.environments:
            environment = environment_library[env_params['model']](**env_params)
            for model_params in self.models:
                model = model_library[model_params['model']](**model_params, state_space=env_params['state_space'],
                                                             action_space=environment.interaction_space,
                                                             start_state=env_params['start_state'])
                for n in tqdm(range(reps), desc=f'Training {model.name} in {environment.name}'):
                    if hasattr(seed, "__getitem__"):
                        np.random.seed(seed[n])
                    else:
                        np.random.seed(seed)
                    res = self.simulate(steps, deepcopy(model), deepcopy(environment))
                    self.results[env_params['name']][model_params['name']].append(res)
        return self.results

    def simulate(self, steps, model, environment):
        results = dict()
        results['states'] = []
        results['actions'] = []
        results['rewards'] = []
        results['attempts'] = []
        results['cumulative'] = []
        results['rolling'] = []
        results['weights']: Dict[str, np.array] = {k: np.zeros((v.shape+(steps,))) for k, v in model.get_weights().items()}
        results['probabilities'] = np.zeros((model.state_space+model.action_space+(steps,)))
        for n in range(steps):
            action = model.act()
            new_state, reward = environment.interact(action)
            model.update(new_state, action, reward)
            results['states'].append(new_state)
            results['actions'].append(action)
            results['rewards'].append(reward)
            results['cumulative'].append(sum(results['rewards']))
            roll = 100
            results['rolling'].append(sum(results['rewards'][max(n-roll, 0):n])/(min(roll, n)+1))
            for k, v in model.get_weights().items():
                results['weights'][k][..., n] = v
            results['probabilities'][..., n] = model.get_probabilities()
            if environment.at_terminal():
                environment.restart()
                model.restart()
                results['attempts'].append(1)
            else:
                results['attempts'].append(0)
        environment.restart()
        model.restart()
        return results

    def get_predictions(self):
        return {env.name:
                {model.name: model.get_predictions() for model in self.models}
                for env in self.environments}

    def update_model_parameters(self, param_dict):
        for model_params in self.models:
            if 'model' in model_params and 'model' in param_dict.keys():
                if model_params['model'] != param_dict['model']:
                    break
            elif 'name' in model_params and 'name' in param_dict.keys():
                if model_params['name'] != param_dict['name']:
                    break
            elif 'model' in param_dict.keys() or 'name' in param_dict.keys():
                raise UserWarning('Model dictionaries supplied to simulator lack name/model specifiers')
            for param in param_dict:
                if param not in model_params:
                    raise UserWarning('New parameters are being added to model dictionaries')
            model_params.update(param_dict)

