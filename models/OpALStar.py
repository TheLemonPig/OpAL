import numpy as np

from models.model import BaseRL
from utils import safe_softmax
from scipy.stats import beta as beta_dist


class OpALStar(BaseRL):

    def __init__(self, action_space, state_space, start_state, alpha_c, alpha_g, alpha_n, beta, gamma, gamma_h, rho, phi, k,
                 r_mag=1, l_mag=-1, T=100, anneal_method='', hs=True, name=None, **kwargs):
        BaseRL.__init__(self, action_space=action_space, state_space=state_space, start_state=start_state, name=name)
        self.alpha_c = alpha_c
        self.alpha_g = alpha_g
        self.alpha_n = alpha_n
        self.beta = beta
        self.gamma = gamma
        self.gamma_h = gamma_h
        self.rho = rho
        self.phi = phi
        self.k = k
        self.T = T
        self.anneal_method = anneal_method
        self.visitation_counter = np.zeros(state_space+action_space)
        # self.qs = np.ones(state_space+action_space) * 0.5
        # self.vs = np.ones(state_space) * 0.5
        self.hs = np.ones(action_space) * 0.5
        self.use_hs = hs
        self.qs = np.ones(action_space) * 0.5
        self.gs = np.ones(state_space+action_space) * 1.0
        self.ns = np.ones(state_space+action_space) * 1.0
        self.eta_c = 1
        self.gamma_c = 1
        self.anneal = 1
        self.r_mag = r_mag
        self.l_mag = l_mag
        self.action = None
        if self.anneal_method == 'variance' or self.anneal_method == 'qs':
            _, var = beta_dist.stats(self.eta_c, self.gamma_c, moments='mv')
            self.anneal = 1/(1+1/(self.T*var))
        

    def act(self):
        self.visitation_counter[self.state] += 1
        beta_g = self.beta*np.max([0, (1+self.rho)])
        beta_n = self.beta*np.max([0, (1-self.rho)])
        net = beta_g * self.gs[self.state] - beta_n * self.ns[self.state]
        if self.anneal_method == 'qs':
            w = self.anneal
            net = net * (1-w) + self.qs[self.state] * w
        p_values = safe_softmax(net)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        self.action = action
        return action

    # TODO: Put model responsible for restarting task

    def update(self, new_state, action, reward):
        delta = self.update_critic(new_state, action, reward)
        self.update_actor(action, delta)
        self.update_metacritic(reward)
        self.state = new_state

    def update_metacritic(self, reward):
        # TODO: Build into code that agent knows if it is a terminal
        if not reward == -0.04:
            if reward >= 0:
                self.eta_c += 1
            else:
                self.gamma_c += 1
            # self.eta_c += reward
            # self.gamma_c += 1 - reward
            # self.eta_c += reward - self.l_mag
            # self.gamma_c += self.r_mag - reward
        n_choices = self.qs.shape[0]
        mean, var = beta_dist.stats(self.eta_c/n_choices, self.gamma_c/n_choices, moments='mv')
        std = np.sqrt(var)
        S = int(mean - self.phi * std > 0.5 or mean + self.phi * std < 0.5)
        self.rho = S * (mean - 0.5) * self.k
        if self.anneal_method == 'variance' or self.anneal_method == 'qs':
            self.anneal = 1/(1+1/(self.T*var))
        elif self.anneal_method == 'visitation':
            self.anneal = 1/(self.visitation_counter[self.state][self.action])
        else:
            self.anneal = 1

    def update_critic(self, new_state, action, reward):
        # Determining likelihood of continuous value being from a certain mean
        # EV[t] = reward[t] + reward[t-1]*gamma_h + reward[t-2]*gamma_h**2 + reward[t-3]*gamma_h**3 + ...
        # This is the same recursive formula used for TD Learning, which leads us to...
        # EV[t] = reward[t] *  EV[t-1]*gamma_h
        # Over time this will tend towards ...
        # EV[t] = mean_reward * 1/(1-gamma_h)
        # But we are interested the trialwise expected value (mean_reward) rather than the accuulated reward, so we add a term...
        # E(reward)[t] = EV[t] * (1-gamma_h) = mean_reward * 1/(1-gamma_h) * (1-gamma_h) = mean_reward
        # delta_h = ...
        # = new_E - old_E...
        # = reward * (1 - gamma_h) + old_EV * gamma_h - old_EV
        # = reward * (1 - gamma_h) + old_EV * (gamma_h - 1)
        # = reward * (1 - gamma_h) - old_EV * (1 - gamma_h)
        # = (reward - old_E) * (1 - gamma_h)
        # So gamma_h can be thought of as stabilizing the actors by acting like a learning rate for the critic
        if self.use_hs:
            old_EV = self.hs[action]
            new_EV = reward * (1 - self.gamma_h) + old_EV * self.gamma_h
            self.hs[action] = new_EV

            # delta_h = (new_EV - old_EV) * (1 - self.gamma_h)
            delta_h = (new_EV - self.qs[action])
            delta = delta_h + self.qs.max() * self.gamma
        else:
            delta = reward - self.qs[action] + self.qs.max() * self.gamma
        # States
        # delta = reward - self.vs[self.state] + self.vs[new_state] * self.gamma
        # self.vs[self.state] += self.alpha_c * delta
        # # State-actions
        # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        # self.qs[self.state][action] += self.alpha_c * delta
        # Actions
        self.qs[action] += self.alpha_c * delta
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
        return {"qs": self.qs, "gs": self.gs, "ns": self.ns}

    def get_probabilities(self):
        beta_g = self.beta * np.max([0, (1 + self.rho)])
        beta_n = self.beta * np.max([0, (1 - self.rho)])
        net = beta_g * self.gs - beta_n * self.ns
        if self.anneal_method == 'qs':
            w = 1 / (1 + np.mean(abs(net)))
            net = net * (1 - w) + self.qs[self.state] * w
        p_values = safe_softmax(net)
        return p_values

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

    def reinitialize_weights(self):
        self.qs = np.ones_like(self.qs) * 0.5
        self.gs = np.ones_like(self.gs)
        self.ns = np.ones_like(self.ns)
