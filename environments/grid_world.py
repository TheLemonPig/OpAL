from environments.environment import BaseEnvironment
import numpy as np


class GridWorld(BaseEnvironment):

    def __init__(self, world_array, start=None):
        BaseEnvironment.__init__(self)
        self.world_array = world_array
        self._start = (0, 0) if start is None else start

    def interact(self, state, interaction):
        new_state = self.move(state, interaction)
        if not self.in_bounds(new_state):
            new_state = state
        reward = self.world_array[new_state]
        return new_state, reward

    @staticmethod
    def move(state, interaction):
        return state[0] + interaction[0], state[1] + interaction[1]

    def get_start(self):
        return self._start

    def in_bounds(self, new_state):
        for i in range(len(new_state)):
            if new_state[i] < 0 or new_state[i] >= self.world_array.shape[i]:
                return False
        return True
