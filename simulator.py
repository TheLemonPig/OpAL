from tqdm import tqdm
from copy import deepcopy

from models import model_library
from environments import environment_library


class Simulator:

    def __init__(self, environments, models):
        self.environments = environments
        self.models = models
        self.results = {env['name']:
                        {model['name']: [] for model in models}
                        for env in environments}

    def run(self, reps, steps):
        for env_params in self.environments:
            environment = environment_library[env_params['name']](**env_params)
            for model_params in self.models:
                model = model_library[model_params['name']](**model_params, state_space=env_params['state_space'],
                                                            action_space=environment.interaction_space,
                                                            start_state=env_params['start_state'])
                for _ in tqdm(range(reps)):
                    res = self.simulate(steps, deepcopy(model), deepcopy(environment))
                    self.results[env_params['name']][model_params['name']].append(res)
        return self.results

    def simulate(self, steps, model, environment):
        results = dict()
        results['states'] = []
        results['actions'] = []
        results['rewards'] = []
        results['attempts'] = []
        for _ in tqdm(range(steps)):
            action = model.act()
            new_state, reward = environment.interact(action)
            model.update(new_state, action, reward)
            results['states'].append(new_state)
            results['actions'].append(action)
            results['rewards'].append(reward)
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

