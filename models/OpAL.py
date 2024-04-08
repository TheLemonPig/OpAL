import numpy as np

from models.model import BaseRL
from utils import safe_softmax


# TODO: Try out having two utility matrices for N and G respectively
# TODO: Add Beta matrix to see if that addresses the issue with values being affected by visitation frequency
# TODO: Set all models with utility matrices of value zero --> do not assume positive bias
# TODO: Experiment with different worlds to learn how each model acts - e.g. OpAL avoids loss more than actor-critic


class OpAL(BaseRL):

    def __init__(self, action_space, state_space, start_state, alpha_c, alpha_g, alpha_n, beta, gamma, rho=0, name=None,
                 **kwargs):
        BaseRL.__init__(self, action_space=action_space, state_space=state_space, start_state=start_state, name=name)
        self.alpha_c = alpha_c
        self.alpha_g = alpha_g
        self.alpha_n = alpha_n
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.qs = np.ones(state_space+action_space) * 0.5
        self.gs = np.ones(state_space+action_space) * 1.0
        self.ns = np.ones(state_space+action_space) * 1.0

    def act(self):
        beta_g = self.beta * (1+self.rho)
        beta_n = self.beta * (1-self.rho)
        net = beta_g * self.gs[self.state] - beta_n * self.ns[self.state]
        p_values = safe_softmax(net)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return action

    # TODO: Put model responsible for restarting task

    def update(self, new_state, action, reward):
        delta = self.update_critic(new_state, action, reward)
        self.update_actor(action, delta)
        self.state = new_state

    def update_critic(self, new_state, action, reward):
        # States
        delta = reward - self.qs[self.state][0] + self.qs[new_state][0] * self.gamma
        self.qs[self.state][0] += self.alpha_c * delta
        # # State-actions
        # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        # self.qs[self.state][action] += self.alpha_c * delta
        # # Actions
        # delta = reward - self.qs[(0, 0)][action] + self.qs[(0, 0)].max() * self.gamma
        # self.qs[(0, 0)][action] += self.alpha_c * delta
        # # Mix-up
        # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        # self.qs[self.state] += self.alpha_c * delta
        return delta

    def update_actor(self, action, delta):
        self.gs[self.state][action] += self.alpha_g * delta * self.gs[self.state][action]
        self.ns[self.state][action] += self.alpha_n * -delta * self.ns[self.state][action]

    def get_weights(self):
        return {"qs": self.qs, "gs": self.gs, "ns": self.ns}

    def get_optimal_policy(self):
        return (self.gs - self.ns).argmax(axis=-1)


