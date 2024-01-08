from environments.environment import BaseEnvironment
import numpy as np

class GridWorld(BaseEnvironment):

    def __init__(self, world_array, start=None):
        BaseEnvironment.__init__(self)
        self.world_array = world_array
        self._start = np.zeros((2,), dtype=np.int32)

    def interact(self, state, interaction):
        # try:
        new_state, reward = self.move(state, interaction)

        # except
        # new_state = state
        return new_state, reward

    def move(self, state, interaction):
        new_state = state + interaction
        if not self.in_bounds(new_state):
            new_state = state
        reward = self.world_array[tuple(new_state)]
        return new_state, reward

    def get_start(self):
        return self._start

    def in_bounds(self, new_state):
        for i in range(len(new_state.shape)+1):
            if new_state[i] < 0 or new_state[i] >= self.world_array.shape[i]:
                return False
        return True
