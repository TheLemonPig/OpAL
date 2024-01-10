from typing import List
import numpy as np

from environments.environment import BaseEnvironment


class BanditTask(BaseEnvironment):

    def __init__(self, ps: np.array, start=None):
        BaseEnvironment.__init__(self)
        self.actions = ps.reshape((1, -1))
        self._start = np.zeros((1, ), dtype=np.int32) if start is None else start

    def interact(self, state, action):
        p = self.actions[state][action]
        return self._start, np.random.choice(2, 1, p=[1-p, p]).item()

    def get_model(self):
        return len(self.actions)

    def get_start(self):
        return self._start
