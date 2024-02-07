from tqdm import tqdm
from copy import deepcopy


class Simulator:

    def __init__(self, model, environment, interaction_space, results=None, max_steps=100, **kwargs):
        self.model = model
        self.model_cache = deepcopy(model)
        self.start_ = self.model.state
        self.environment = environment
        self.interaction_space = interaction_space
        self.terminals = self.environment.terminals
        self.results = {'States': []} if results is None else {result: None for result in results}
        self.n_steps = 0
        self.max_steps = max_steps

    def simulate(self, steps):
        self.reset()
        for _ in tqdm(range(steps)):
            self.step()
        return self.get_results()

    def step(self):
        state, action = self.model.act()
        interaction = self.interaction_space[action]
        new_state, reward = self.environment.interact(state, interaction)
        self.model.update(new_state, action, reward)
        if self.terminals and (new_state in self.terminals) or self.n_steps > self.max_steps:
            # reinitialize RL path
            self.model.state = self.start_
            self.n_steps = 0
        else:
            self.n_steps += 1
        self.results['rewards'].append(reward)
        self.results['states'].append(new_state)
        return reward, new_state

    def get_results(self):
        return self.results

    def get_predictions(self):
        return self.model.get_predictions()

    def reset(self):
        self.model = deepcopy(self.model_cache)
        self.n_steps = 0
        self.results['rewards'] = []
        self.results['states'] = []
