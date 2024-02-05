import numpy as np

from models.model import BaseRL
from utils import safe_softmax
from scipy.stats import beta as beta_dist


class OpALStar(BaseRL):

    def __init__(self, actions, domain_shape, state, alpha_c, alpha_g, alpha_n, beta, gamma, rho, phi, k,
                 r_mag=1, l_mag=-1, T=100, use_qs=True):
        BaseRL.__init__(self, domain=domain_shape, state=state)
        self.alpha_c = alpha_c
        self.alpha_g = alpha_g
        self.alpha_n = alpha_n
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.phi = phi
        self.k = k
        self.T = T
        actor_shape = list(domain_shape) + [len(actions)]
        self.use_qs = use_qs
        self.state = state
        self.qs = np.ones(actor_shape) * 0.5
        self.gs = np.ones(actor_shape) * 1.0
        self.ns = np.ones(actor_shape) * 1.0
        self.eta_c = 1
        self.gamma_c = 1
        self.anneal = 1
        self.r_mag = r_mag
        self.l_mag = l_mag

    def act(self):
        beta_g = self.beta*np.max([0, (1+self.rho)])
        beta_n = self.beta*np.max([0, (1-self.rho)])
        net = beta_g * self.gs[self.state] - beta_n * self.ns[self.state]
        if self.use_qs:
            net += self.qs[self.state]  # should I add a weight that only elevates qs when gs/ns converge to zero?
        p_values = safe_softmax(net)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return self.state, action

    def update(self, new_state, action, reward):
        self.update_metacritic(reward)
        delta = self.update_critic(new_state, action, reward)
        self.update_actor(action, delta)
        self.state = new_state

    def update_metacritic(self, reward):
        self.eta_c += reward - self.l_mag
        self.gamma_c += self.r_mag - reward
        mean, var = beta_dist.stats(self.eta_c, self.gamma_c, moments='mv')
        std = np.sqrt(var)
        S = int(mean - self.phi * std > 0.5 or mean + self.phi * std < 0.5)
        self.rho = S * (mean - 0.5) * self.k
        self.anneal = 1/(1+1/(self.T*var))
        # self.anneal = 1

    def update_critic(self, new_state, action, reward):
        delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        self.qs[self.state] += self.alpha_c * delta
        return delta

    def update_actor(self, action, delta):
        alpha_gt = self.alpha_g * self.anneal
        alpha_nt = self.alpha_n * self.anneal
        self.gs[self.state][action] += alpha_gt * self.f(delta) * self.gs[self.state][action]
        self.ns[self.state][action] += alpha_nt * self.f(-delta) * self.ns[self.state][action]

    def f(self, delta):
        return delta/(self.r_mag-self.l_mag)

    def get_predictions(self):
        return {"qs": self.qs, "gs": self.gs, "ns": self.ns}

    def get_optimal_policy(self):
        return (self.gs - self.ns).argmax(axis=-1)
