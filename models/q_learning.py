import numpy as np

from models.model import BaseRL
from utils import tempered_softmax


class QLearning(BaseRL):

    def __init__(self, n_options, state, lr, temperature, steps):
        BaseRL.__init__(self, n_options, state)
        # parameters
        self.lr = lr
        self.temperature = temperature
        # arrays
        self.qs = np.ones((steps+1, n_options)) * 0.5
        self.time = 0

    def act(self):
        p_values = tempered_softmax(self.qs[self.time], self.temperature)
        choice = np.random.choice(len(p_values), 1, p=p_values)[0]
        self.time += 1
        return choice

    def update(self, choice, reward):
        delta = reward - self.qs[self.time-1, choice]
        self.qs[self.time] = self.qs[self.time-1]
        self.qs[self.time, choice] += self.lr * delta

    def get_predictions(self):
        return self.qs[-1]


