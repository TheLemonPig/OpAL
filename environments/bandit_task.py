from typing import List
import numpy as np

from environments.environment import BaseEnvironment


class BanditTask(BaseEnvironment):

    def __init__(self, interactions, state_space, start_state, deterministic=False, ps=None, name=None,
                 **kwargs):
        BaseEnvironment.__init__(self, interactions=interactions, state_space=state_space, start_state=start_state,
                                 name=name)
        self.deterministic = deterministic
        self.actions = np.array(ps).reshape((1, -1))

    def interact(self, action):
        p = self.actions[self.model_state][action]
        if self.deterministic:
            reward = p
        else:
            reward = np.random.choice(2, 1, p=[1-p, p]).item()
        return self.model_state, reward
