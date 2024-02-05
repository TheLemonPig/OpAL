from environments.environment import BaseEnvironment
import numpy as np


class GridWorld(BaseEnvironment):

    def __init__(self, world_array, terminals=None):
        BaseEnvironment.__init__(self)
        self.world_array = world_array
        self.terminals = terminals if terminals else dict()

    def interact(self, state, interaction):
        new_state = self.move(state, interaction)
        if not self.in_bounds(new_state):
            new_state = state
        reward = self.sample(new_state)
        return new_state, reward

    def sample(self, new_state):
        if new_state in self.terminals.keys():
            # grid world value modeled as bandit probability
            p = abs(self.world_array[new_state])
            sign = self.world_array[new_state]/abs(self.world_array[new_state])
            reward = np.random.choice(2, 1, p=[1 - p, p]).item() * sign
        else:
            # grid world value modeled as reward
            reward = self.world_array[new_state]
        return reward

    @staticmethod
    def move(state, interaction):
        return state[0] + interaction[0], state[1] + interaction[1]

    def in_bounds(self, new_state):
        for i in range(len(new_state)):
            if new_state[i] < 0 or new_state[i] >= self.world_array.shape[i]:
                return False
        return True
