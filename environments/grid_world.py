from environments.environment import BaseEnvironment
import numpy as np


class GridWorld(BaseEnvironment):

    def __init__(self, interactions, state_space, start_state, deterministic=True, terminal_states=None,
                 non_terminal_loss=0, name=None, **kwargs):
        BaseEnvironment.__init__(self, interactions=interactions, state_space=state_space,
                                 start_state=start_state, name=name)
        self.world_array = np.ones(state_space) * non_terminal_loss
        self.non_terminal_loss = non_terminal_loss
        for terminal_location in terminal_states.keys():
            self.world_array[terminal_location] = terminal_states[terminal_location]
        self.terminal_states = terminal_states
        self.deterministic = deterministic

    def interact(self, action):
        interaction = self.interactions[action]
        new_state = self.translate(self.model_state, interaction)
        if self.in_bounds(new_state):
            self.model_state = new_state
        else:
            new_state = self.model_state
        reward = self.sample(new_state)
        return new_state, reward

    def sample(self, new_state):
        if self.at_terminal() and not self.deterministic:
            # grid world value modeled as bandit probability
            p = abs(self.world_array[new_state])
            sign = self.world_array[new_state]/abs(self.world_array[new_state])
            reward = np.random.choice(2, 1, p=[1 - p, p]).item() * sign
        else:
            # grid world value modeled as reward
            reward = self.world_array[new_state]
        return reward

    @staticmethod
    def translate(state, interaction):
        return state[0] + interaction[0], state[1] + interaction[1]

    def in_bounds(self, new_state):
        for i in range(len(new_state)):
            if new_state[i] < 0 or new_state[i] >= self.world_array.shape[i]:
                return False
        return True

    def at_terminal(self):
        if self.terminal_states and self.model_state in self.terminal_states:
            return True
        else:
            return False
