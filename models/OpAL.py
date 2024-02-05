import numpy as np

from models.model import BaseRL
from utils import safe_softmax


# TODO: Try out having two utility matrices for N and G respectively
# TODO: Add Beta matrix to see if that addresses the issue with values being affected by visitation frequency
# TODO: Set all models with utility matrices of value zero --> do not assume positive bias
# TODO: Experiment with different worlds to learn how each model acts - e.g. OpAL avoids loss more than actor-critic


class OpAL(BaseRL):

    def __init__(self, actions, domain_shape, state, alpha_c, alpha_g, alpha_n, beta, gamma, rho=0, use_state=True):
        BaseRL.__init__(self, domain=domain_shape, state=state)
        self.alpha_c = alpha_c
        self.alpha_g = alpha_g
        self.alpha_n = alpha_n
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        actor_shape = list(domain_shape) + [len(actions)]
        self.use_state = use_state
        self.state = state
        self.qs = np.ones(actor_shape) * 0.5
        self.gs = np.ones(actor_shape) * 1.0
        self.ns = np.ones(actor_shape) * 1.0

    def act(self):
        beta_g = self.beta * (1+self.rho)
        beta_n = self.beta * (1-self.rho)
        net = beta_g * self.gs[self.state] - beta_n * self.ns[self.state]
        p_values = safe_softmax(net)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return self.state, action

    def update(self, new_state, action, reward):
        delta = self.update_critic(new_state, action, reward)
        self.update_actor(action, delta)
        self.state = new_state

    def update_critic(self, new_state, action, reward):
        delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        self.qs[self.state][action] += self.alpha_c * delta
        return delta

    def update_actor(self, action, delta):
        self.gs[self.state][action] += self.alpha_g * delta * self.gs[self.state][action]
        self.ns[self.state][action] += self.alpha_n * -delta * self.ns[self.state][action]

    def get_predictions(self):
        return {"qs": self.qs, "gs": self.gs, "ns": self.ns}

    def get_optimal_policy(self):
        return (self.gs - self.ns).argmax(axis=-1)


