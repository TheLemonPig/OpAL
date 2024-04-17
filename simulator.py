from tqdm import tqdm
from copy import deepcopy
import numpy as np

from models import model_library
from environments import environment_library


class Simulator:

    def __init__(self, environments, models):
        self.environments = environments
        self.models = models
        self.results = {env['name']:
                        {model['name']: [] for model in models}
                        for env in environments}

    def run(self, reps, steps, seed=None):
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
                    res = self.simulate(steps, deepcopy(model), deepcopy(environment))
                    self.results[env_params['name']][model_params['name']].append(res)
        return self.results

    def simulate(self, steps, model, environment):
        results = dict()
        results['states'] = []
        results['new_states'] = []
        results['actions'] = []
        results['rewards'] = []
        results['attempts'] = []
        results['cumulative'] = []
        results['rolling'] = []
        results['rho'] = []
        n_steps = 0
        # for n in tqdm(range(steps)):
        for n in range(steps):
            if n % 50 == 0:
                x = 0
            action = model.act()
            results['states'].append(model.state)
            new_state, reward = environment.interact(action)
            model.update(new_state, action, reward)
            results['new_states'].append(new_state)
            results['actions'].append(action)
            results['rewards'].append(reward)
            results['cumulative'].append(sum(results['rewards']))
            roll = steps // 10
            results['rolling'].append(sum(results['rewards'][max(n-roll, 0):n])/(min(roll, n)+1))
            if model.name.startswith('OpAL'):
                results['rho'] = model.rho
            if environment.at_terminal() or environment.time_up(n_steps):
                environment.restart()
                model.restart()
                n_steps = 0
                results['attempts'].append(1)
            else:
                n_steps += 1
                results['attempts'].append(0)

        environment.restart()
        model.restart()
        return results

    def get_predictions(self):
        return {env.name:
                {model.name: model.get_predictions() for model in self.models}
                for env in self.environments}

