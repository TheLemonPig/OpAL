from tqdm import tqdm
from copy import deepcopy
import numpy as np
from typing import Dict

from models import model_library
from environments import environment_library


class Simulator:

    def __init__(self, environments, models):
        self.environments = environments
        self.models = models
        self.results = {env['name']:
                        {model['name']: [] for model in models}
                        for env in environments}

    def run(self, reps, steps, seed=None, thin=1):
        for env_params in self.environments:
            environment = environment_library[env_params['model']](**env_params)
            for model_params in self.models:
                model = model_library[model_params['model']](**model_params, state_space=env_params['state_space'],
                                                            action_space=environment.interaction_space,
                                                            start_state=env_params['start_state'])
                # for n in tqdm(range(reps)):
                for n in tqdm(range(reps), desc=f'Training {model.name} in {environment.name}'):
                    if hasattr(seed, "__getitem__"):
                        np.random.seed(seed[n])
                    else:
                        np.random.seed(seed)
                    res = self.simulate(steps, deepcopy(model), deepcopy(environment), thin=thin)
                    self.results[env_params['name']][model_params['name']].append(res)
        return self.results

    def simulate(self, steps, model, environment, thin=1):
        results = dict()
        results['states'] = []
        results['new_states'] = []
        results['actions'] = []
        results['rewards'] = []
        results['attempts'] = []
        results['cumulative'] = []
        results['rolling'] = []
        results['rho'] = []
        results['anneal'] = []
        results['weights']: Dict[str, np.array] = {k: np.zeros((v.shape + (steps,))) for k, v in
                                                   model.get_weights().items()}
        results['probabilities'] = np.zeros((model.state_space + model.action_space + (steps,)))
        n_steps = 0
        # for n in tqdm(range(steps)):
        for n in range(steps):
            if n % 50 == 0:
                x = 0
            action = model.act()
            if n % thin == 0:
                results['states'].append(model.state)
            new_state, reward = environment.interact(action)
            model.update(new_state, action, reward)
            if n % thin == 0:
                results['new_states'].append(new_state)
                results['actions'].append(action)
                results['rewards'].append(reward)
                results['cumulative'].append(sum(results['rewards']))
                roll = steps // 10
                results['rolling'].append(sum(results['rewards'][max(n-roll, 0):n])/(min(roll, n)+1))
                if model.name.startswith('OpAL'):
                    results['rho'].append(model.rho)
                    results['anneal'].append(model.anneal)
                for k, v in model.get_weights().items():
                    results['weights'][k][..., n] = v
                results['probabilities'][..., n] = model.get_probabilities()
            if environment.at_terminal() or environment.time_up(n_steps):
                environment.restart()
                model.restart()
                n_steps = 0
                if n % thin == 0:
                    results['attempts'].append(1)
            else:
                n_steps += 1
                if n % thin == 0:
                    results['attempts'].append(0)

        environment.restart()
        model.restart()
        return results

    def get_predictions(self):
        return {env.name:
                {model.name: model.get_predictions() for model in self.models}
                for env in self.environments}

