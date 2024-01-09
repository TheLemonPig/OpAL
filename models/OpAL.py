import numpy as np

from models.model import BaseRL
from utils import tempered_softmax


class OpAL(BaseRL):

    def __init__(self, actions, domain_shape, state, lr_g, lr_n, temperature):
        BaseRL.__init__(self, domain=domain_shape, state=state)
        self.lr_g = lr_g
        self.lr_n = lr_n
        self.temperature = temperature
        v_shape = list(domain_shape) + [len(actions)]
        self.state = state
        self.gs = np.ones(v_shape) * 0.
        self.ns = np.ones(v_shape) * 0.

    def act(self):
        net_vs = self.gs[tuple(self.state)] - self.ns[tuple(self.state)]
        if (net_vs > 10).sum() > 0:
            net_vs[net_vs > 10] = 10
        p_values = tempered_softmax(net_vs, self.temperature)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return self.state, action

    def update(self, new_state, action, reward):
        delta_g = reward - self.gs[tuple(new_state)][action]
        delta_n = reward - self.ns[tuple(new_state)][action]
        self.gs[tuple(new_state)][action] += self.lr_g * delta_g
        self.ns[tuple(new_state)][action] += self.lr_n * -delta_n
        self.state = new_state

    def get_predictions(self):
        return {"gs": self.gs, "ns": self.ns}


