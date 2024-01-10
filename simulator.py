from tqdm import tqdm


class Simulator:

    def __init__(self, model, environment, interaction_space, terminals=None):
        self.model = model
        self.environment = environment
        self.interaction_space = interaction_space
        self.terminals = {self.model.state: 0} if terminals is None else terminals
        self.net_reward = 0

    def simulate(self, steps):
        total_reward = 0
        for _ in tqdm(range(steps)):
            total_reward += self.step()
        self.net_reward = total_reward / steps
        return self.get_results()

    def step(self):
        state, action = self.model.act()
        interaction = self.interaction_space[action]
        new_state, reward = self.environment.interact(state, interaction)
        self.model.update(new_state, action, reward)
        if self.terminals and new_state in self.terminals:
            # reinitialize RL path
            self.model.state = self.environment.get_start()
        return reward

    def get_results(self):
        return self.model.get_predictions(), self.net_reward
