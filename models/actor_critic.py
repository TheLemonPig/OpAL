import numpy as np

from models.model import BaseRL
from utils import safe_softmax


class ActorCritic(BaseRL):

    def __init__(self, n_actions, alpha, beta, gamma=0, **kwargs):
        BaseRL.__init__(self, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # self.location_counter = np.zeros(domain)
        self.vs = np.ones(self.domain) * 0.5
        p_shape = list(self.domain) + [n_actions]
        self.ps = np.ones(p_shape) * 0.5

    def act(self):
        p_values = safe_softmax(self.ps[self.state] * self.beta)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return self.state, action

    def update(self, new_state, action, reward):
        delta = self.update_critic(new_state, reward)
        self.update_actor(action, delta)
        self.state = new_state

    def update_critic(self, new_state, reward):
        delta = reward - self.vs[self.state] + self.vs[new_state] * self.gamma
        self.vs[self.state] += self.alpha * delta
        return delta

    def update_actor(self, action, delta):
        self.ps[self.state][action] += self.beta * delta

    def get_predictions(self):
        return {"ps": self.ps, "vs": self.vs}

    def get_optimal_policy(self):
        return self.ps.argmax(axis=-1)

