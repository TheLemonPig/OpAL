import numpy as np


# Basic Template for all RL models
class BaseRL:

    def __init__(self, action_space, state_space, start_state, name=None):
        self.action_space = action_space
        self.state_space = state_space
        self.name = "NamelessModel" if name is None else name
        self.start_state = start_state
        self.state = start_state
    #     self.memory = [(state, None) for _ in range(memory)]
    #     self.mod_count = 0
    #
    # def store(self, state, action):
    #     self.memory[self.mod_count] = (state, action)
    #     self.mod_count = (self.mod_count + 1) % len(self.memory)
    #
    # def retrieve(self, t_n):
    #     return self.memory[self.mod_count - t_minus_n]
    # for t in range(self.memory):
    #     state_t, action_t = self.retrieve(t)
    #     self.qs[state_t][action_t] += self.lr * self.gamma * self.qs[new_state][action]

    def act(self):
        ...

    # TODO: Put model responsible for restarting task

    def update(self, *args):
        ...

    def get_weights(self):
        ...

    def get_optimal_policy(self):
        ...

    def restart(self):
        self.state = self.start_state
