from tqdm import tqdm


class Simulator:

    def __init__(self, model, environment, interaction_space):
        self.model = model
        self.environment = environment
        self.interaction_space = interaction_space
        self.results = None

    def simulate(self, steps):
        for _ in tqdm(range(steps)):
            self.step()
        return self.get_results()

    def step(self):
        state, action = self.model.act()
        interaction = self.interaction_space[action]
        new_state, reward = self.environment.interact(state, interaction)
        self.model.update(new_state, action, reward)

    def get_results(self):
        return self.model.get_predictions()
