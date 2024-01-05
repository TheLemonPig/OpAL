from typing import List
import numpy as np

from environments.environment import BaseEnvironment


class BanditTask(BaseEnvironment):

    def __init__(self, ps: List):
        BaseEnvironment.__init__(self)
        self.ps = ps

    def sample(self, choice):
        p = self.ps[choice]
        return np.random.choice(2, 1, p=[1-p, p])[0]

    def get_domain(self):
        return len(self.ps)
