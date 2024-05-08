import numpy as np

from models.model import BaseRL
from utils import safe_softmax
from scipy.stats import beta as beta_dist


class OpALPlus(BaseRL):

    def __init__(self, action_space, state_space, start_state, alpha_c, alpha_g, alpha_n, beta, gamma, rho,
                 r_mag=1, l_mag=-1, T=100, anneal_method='variance', name=None, **kwargs):
        BaseRL.__init__(self, action_space=action_space, state_space=state_space, start_state=start_state, name=name)
        self.alpha_c = alpha_c
        self.alpha_g = alpha_g
        self.alpha_n = alpha_n
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.T = T
        self.anneal_method = anneal_method
        self.visitation_counter = np.zeros(state_space+action_space)
        self.vs = np.ones(state_space) * 0.5
        # self.qs = np.ones(state_space + action_space) * 0.5
        self.gs = np.ones(state_space+action_space) * 1.0
        self.ns = np.ones(state_space+action_space) * 1.0
        self.eta_c = 1
        self.gamma_c = 1
        self.anneal = 1
        self.r_mag = r_mag
        self.l_mag = l_mag
        self.action = None

    def act(self):
        self.visitation_counter[self.state] += 1
        beta_g = self.beta*np.max([0, (1+self.rho)])
        beta_n = self.beta*np.max([0, (1-self.rho)])
        net = beta_g * self.gs[self.state] - beta_n * self.ns[self.state]
        # if self.anneal_method == 'qs':
        #     w = 1/(1+np.mean(abs(net)))
        #     net = net * (1-w) + self.vs[self.state] * w
        p_values = safe_softmax(net)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        self.action = action
        return action

    # TODO: Put model responsible for restarting task

    def update(self, new_state, action, reward):
        self.update_metacritic(reward)
        delta = self.update_critic(new_state, action, reward)
        self.update_actor(action, delta)
        self.state = new_state

    def update_metacritic(self, reward):
        # TODO: Build into code that agent knows if it is a terminal
        if not reward == -0.04:
            self.eta_c += reward - self.l_mag
            self.gamma_c += self.r_mag - reward
        mean, var = beta_dist.stats(self.eta_c, self.gamma_c, moments='mv')
        if self.anneal_method == 'variance':
            self.anneal = 1/(1+1/(self.T*var))
        elif self.anneal_method == 'visitation':
            self.anneal = 1/(self.visitation_counter[self.state][self.action])
        else:
            self.anneal = 1

    def update_critic(self, new_state, action, reward):
        # States
        delta = reward - self.vs[self.state] + self.vs[new_state] * self.gamma
        self.vs[self.state] += self.alpha_c * delta
        # # State-actions
        # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        # self.qs[self.state][action] += self.alpha_c * delta
        # Actions
        # delta = reward - self.qs[(0, 0)][action] + self.qs[(0, 0)].max() * self.gamma
        # self.qs[(0, 0)][action] += self.alpha_c * delta
        # # Mix-up
        # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        # self.qs[self.state] += self.alpha_c * delta
        return delta

    def update_actor(self, action, delta):
        alpha_gt = self.alpha_g * self.anneal
        alpha_nt = self.alpha_n * self.anneal
        self.gs[self.state][action] += alpha_gt * self.f(delta) * self.gs[self.state][action]
        self.ns[self.state][action] += alpha_nt * self.f(-delta) * self.ns[self.state][action]

    def f(self, delta):
        return delta/(self.r_mag-self.l_mag)

    def get_weights(self):
        return {"vs": self.vs, "gs": self.gs, "ns": self.ns}

    def get_optimal_policy(self):
        # This is out of date
        return (self.gs - self.ns).argmax(axis=-1)

    def get_probabilities(self):
        beta_g = self.beta * np.max([0, (1 + self.rho)])
        beta_n = self.beta * np.max([0, (1 - self.rho)])
        net = beta_g * self.gs - beta_n * self.ns
        # if self.anneal_method == 'qs':
        #     w = 1 / (1 + np.mean(abs(net)))
        #     net = net * (1 - w) + self.vs[self.state] * w
        p_values = np.zeros_like(net)
        if len(self.state_space) == 2:
            for i in range(self.state_space[0]):
                for j in range(self.state_space[1]):
                    p_values[i, j] = safe_softmax(net[i, j])
        elif len(self.state_space) == 1:
            for i in range(self.state_space[0]):
                p_values[i] = safe_softmax(net[i])
        else:
            raise ValueError

        return p_values