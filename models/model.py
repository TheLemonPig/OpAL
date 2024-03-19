import numpy as np


# Basic Template for all RL models
class BaseRL:

    def __init__(self, action_space, state_space, start_state, name=None):
        self.action_space = action_space
        self.state_space = state_space
        self.name = "NamelessModel" if name is None else name
        self.start_state = start_state
        self.state = start_state

    def act(self):
        raise NotImplementedError

    # TODO: Put model responsible for restarting task

    def update(self, *args):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def get_optimal_policy(self):
        raise NotImplementedError

    def restart(self):
        self.state = self.start_state
        self.reinitialize_weights()

    def reinitialize_weights(self):
        raise NotImplementedError(f'{self.name}')

    def get_probabilities(self):
        raise NotImplementedError(f'{self.name}')
