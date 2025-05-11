import numpy as np

from models.model import BaseRL
from utils import safe_softmax
from scipy.stats import beta as beta_dist


class OpALStar(BaseRL):

    def __init__(self, action_space, state_space, start_state, alpha_c, alpha_g, alpha_n, beta, gamma, gamma_h, rho, phi, k,
                 r_mag=1, l_mag=-1, T=100, anneal_method='', critic='actions', hs=True, name=None, **kwargs):
        BaseRL.__init__(self, action_space=action_space, state_space=state_space, start_state=start_state, name=name)
        self.critic = critic
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
        self.vs_stateaction = np.ones(state_space+action_space) * 0.5
        self.vs_state = np.ones(state_space) * 0.5
        self.vs_action = np.ones(action_space) * 0.5
        self.history = [[] for _ in range(action_space[0])]
        self.hs = np.ones(action_space) * 0.5
        self.use_hs = hs
        self.gs = np.ones(state_space+action_space) * 1.0
        self.ns = np.ones(state_space+action_space) * 1.0
        self.eta_c = 1
        self.gamma_c = 1
        self.anneal = 1
        self.r_mag = r_mag
        self.l_mag = l_mag
        self.action = None
        self.at_terminal = False
        if self.anneal_method == 'variance' or self.anneal_method == 'qs':
            _, var = beta_dist.stats(self.eta_c, self.gamma_c, moments='mv')
            self.anneal = 1/(1+1/(self.T*var))
        
    def act(self):
        self.visitation_counter[self.state] += 1
        beta_g = self.beta * np.max([0, (1+self.rho)])
        beta_n = self.beta * np.max([0, (1-self.rho)])
        net = beta_g * self.gs[self.state] - beta_n * self.ns[self.state]
        p_values = safe_softmax(net)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        self.action = action
        return action

    # TODO: Put model responsible for restarting task

    def update(self, new_state, action, reward, terminal):
        self.at_terminal = terminal
        delta = self.update_critic(new_state, action, reward)
        self.update_actor(action, delta)
        self.update_metacritic(reward)
        self.state = new_state

    def update_metacritic(self, reward):
        if self.at_terminal:
            if reward >= 0.5:
                self.eta_c += 1
            else:
                self.gamma_c += 1
            # self.eta_c += reward
            # self.gamma_c += 1 - reward
            # self.eta_c += max(0,reward - self.l_mag)
            # self.gamma_c += max(0,self.r_mag - reward)
        n_choices = self.vs_action.shape[0]
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
        if self.critic=="actions":
            # Actions
            delta = reward - self.vs_action[action] + self.vs_action.max() * self.gamma * (not self.at_terminal)
            self.vs_action[action] += self.alpha_c * delta
        elif self.critic=="states":
            # States
            delta = reward - self.vs_state[self.state] + self.vs_state[new_state] * self.gamma * (not self.at_terminal)
            self.vs_state[self.state] += self.alpha_c * delta
        elif self.critic=="state-actions":
            # State-actions
            delta = reward - self.vs_state[self.state][action] + self.vs_state[new_state].max() * self.gamma * (not self.at_terminal)
            self.vs_state[self.state][action] += self.alpha_c * delta
        else:
            raise KeyError('critic type not included\nPlease choose from actions, states, state-actions\n')
        # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma * (not self.at_terminal)
        # self.qs[self.state][action] += self.alpha_c * delta
        return delta

    def update_actor(self, action, delta):
        alpha_gt = self.alpha_g * self.anneal
        alpha_nt = self.alpha_n * self.anneal
        self.gs[self.state][action] += alpha_gt * self.f(delta) * self.gs[self.state][action]
        self.ns[self.state][action] += alpha_nt * self.f(-delta) * self.ns[self.state][action]

    def f(self, delta):
        return delta/(self.r_mag-self.l_mag)

    def get_weights(self):
        if self.critic=="actions":
            return {"vs": self.vs_action, "gs": self.gs, "ns": self.ns}
        if self.critic=="states":
            return {"vs": self.vs_state, "gs": self.gs, "ns": self.ns}
        if self.critic=="state-actions":
            return {"vs": self.vs_stateaction, "gs": self.gs, "ns": self.ns}
        else:
            raise KeyError('critic type not included\nPlease choose from actions, states, state-actions\n')      

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
        if self.critic=="actions":
            self.vs_action = np.ones_like(self.vs_action) * 0.5
        if self.critic=="states":
            self.vs_state = np.ones_like(self.vs_state) * 0.5
        if self.critic=="state-actions":
            self.vs_stateaction = np.ones_like(self.vs_stateaction) * 0.5
        self.hs = np.ones_like(self.hs) * 0.5
        self.gs = np.ones_like(self.gs)
        self.ns = np.ones_like(self.ns)
        self.history = []

    # def update_critic(self, new_state, action, reward):
    #     # Determining likelihood of continuous value being from a certain mean
    #     # EV[t] = reward[t] + reward[t-1]*gamma_h + reward[t-2]*gamma_h**2 + reward[t-3]*gamma_h**3 + ...
    #     # This is the same recursive formula used for TD Learning, which leads us to...
    #     # EV[t] = reward[t] *  EV[t-1]*gamma_h
    #     # Over time this will tend towards ...
    #     # EV[t] = mean_reward * 1/(1-gamma_h)
    #     # But we are interested the trialwise expected value (mean_reward) rather than the accuulated reward, so we add a term...
    #     # E(reward)[t] = EV[t] * (1-gamma_h) = mean_reward * 1/(1-gamma_h) * (1-gamma_h) = mean_reward
    #     # delta_h = ...
    #     # = new_E - old_E...
    #     # = reward * (1 - gamma_h) + old_EV * gamma_h - old_EV
    #     # = reward * (1 - gamma_h) + old_EV * (gamma_h - 1)
    #     # = reward * (1 - gamma_h) - old_EV * (1 - gamma_h)
    #     # = (reward - old_E) * (1 - gamma_h)
    #     # So gamma_h can be thought of as stabilizing the actors by acting like a learning rate for the critic
    #     if self.use_hs:
    #         old_EV = self.hs[action]
    #         action_history = self.history[action]
    #         n_samples = len(action_history)
    #         # mean = np.mean(action_history)  # -- this will probably be what allows us the best of both
    #         # var = np.var(action_history)  # i.e. fast to change at first and then slow to change over time BUT speed up if the EV is clearly wrong

    #         new_EV = (reward + old_EV * self.gamma_h * n_samples) / (n_samples * self.gamma_h + 1)
    #         self.hs[action] = new_EV
    #         self.history[action].append(reward)
    #         # (0.5 * 0.1 + 0.5 * 0.9 * n) / (0.9 * (n+1)) = (0.45n + 0.05)/(0.9*(n+1)) =
    #         # (0.45n + 0.45 - 0.5)/(0.9*(n+1)) = (0.45/0.9) - 0.5/(0.9*(n+1)) = 0.5 - 
    #         # ((m - e) + m * g * n) / (g * n + 1) = (m * (g * n + 1) - e) / (g * n + 1) = m - e/(g*n+1)
    #         # (m + (m - e) * g * n) / (g * n + 1) = (m * (g * n + 1) - e * g * n) / (g * n + 1) = m - e(g*n)/(g*n+1) = m - e(1 - 1/(g*n+1))
    #         # We don't want gamma multiplied to n as n grows unrestricted
    #         # Instead we want g as a multiplier on e, essentially as a learning rate
    #         # delta_h = (new_EV - old_EV) * (1 - self.gamma_h)
    #         # i.e. instead of m - e/(g*n+1) --> m - e*(1-g)/(n+1)
    #         # instead of m - e(1 - 1/(g*n+1)) --> m - e*(1-g)(1 - 1/(n+1))
    #         # Which going backwards implies...two different equations which doesn't work since you can't assume you know what the mean is
    #         delta_h = (new_EV - self.qs[action])
    #         delta = delta_h + self.qs.max() * self.gamma
    #     else:
    #         delta = reward - self.qs[action] + self.qs.max() * self.gamma
    #     # States
    #     # delta = reward - self.vs[self.state] + self.vs[new_state] * self.gamma
    #     # self.vs[self.state] += self.alpha_c * delta
    #     # # State-actions
    #     # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
    #     # self.qs[self.state][action] += self.alpha_c * delta
    #     # Actions
    #     self.qs[action] += self.alpha_c * delta
    #     # # Mix-up
    #     # delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
    #     # self.qs[self.state] += self.alpha_c * delta
    #     return delta