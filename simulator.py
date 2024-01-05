from tqdm import tqdm


class Simulator:

    def __init__(self, model, environment):
        self.model = model
        self.environment = environment
        self.results = None

    def simulate(self, steps):
        for _ in tqdm(range(steps)):
            self.step()
        return self.get_results()

    def step(self):
        choice = self.model.act()
        reward = self.environment.sample(choice)
        self.model.update(choice, reward)

    def get_results(self):
        return self.model.get_predictions()
