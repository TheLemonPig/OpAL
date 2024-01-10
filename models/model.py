import numpy as np


# Basic Template for all RL models
class BaseRL:

    def __init__(self, domain, state):
        self.state = state
        self.domain = domain
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

    def update(self, *args):
        ...

    def get_predictions(self):
        ...
