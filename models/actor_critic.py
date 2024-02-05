import numpy as np

from models.model import BaseRL
from utils import tempered_softmax


class ActorCritic(BaseRL):

    def __init__(self, actions, domain_shape, state, alpha, beta, temperature, gamma=0):
        BaseRL.__init__(self, domain=domain_shape, state=state)
        self.state = state
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.gamma = gamma
        self.vs = np.ones(domain_shape) * 0.5
        p_shape = list(domain_shape) + [len(actions)]
        self.ps = np.ones(p_shape) * 0.5

    def act(self):
        p_values = tempered_softmax(self.ps[self.state], self.temperature)
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

