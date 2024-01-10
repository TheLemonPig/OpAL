import numpy as np

from models.model import BaseRL
from utils import tempered_softmax


class QLearning(BaseRL):

    def __init__(self, actions, domain_shape, state, lr, temperature, gamma=0):
        BaseRL.__init__(self, domain=domain_shape, state=state)
        self.lr = lr
        self.temperature = temperature
        self.gamma = gamma
        q_shape = list(domain_shape) + [len(actions)]
        self.state = state
        # self.qs = np.ones(len(actions)) * 0.5
        self.qs = np.ones(q_shape) * 0.

    def act(self):
        p_values = tempered_softmax(self.qs[self.state], self.temperature)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return self.state, action

    def update(self, new_state, action, reward):
        delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        self.qs[self.state][action] += self.lr * delta
        self.state = new_state

    def get_predictions(self):
        return self.qs


