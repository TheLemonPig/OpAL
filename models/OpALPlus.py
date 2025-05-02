import numpy as np

from models.model import BaseRL
from utils import safe_softmax
from scipy.stats import beta as beta_dist


class OpALPlus(BaseRL):

    def __init__(self, action_space, state_space, start_state, alpha_c, alpha_g, alpha_n, beta, gamma, rho,
                 r_mag=1, l_mag=-1, T=100, anneal_method='variance', critic='actions', name=None, **kwargs):
        BaseRL.__init__(self, action_space=action_space, state_space=state_space, start_state=start_state, name=name)
        self.critic = critic
        self.alpha_c = alpha_c
        self.alpha_g = alpha_g
        self.alpha_n = alpha_n
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.T = T
        self.anneal_method = anneal_method
        self.visitation_counter = np.zeros(state_space+action_space)
        self.vs_stateaction = np.ones(state_space+action_space) * 0.5
        self.vs_state = np.ones(state_space) * 0.5
        self.vs_action = np.ones(action_space) * 0.5
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
            self.eta_c += reward
            self.gamma_c += 1 - reward
        n_choices = self.qs.shape[0]
        mean, var = beta_dist.stats(self.eta_c/n_choices, self.gamma_c/n_choices, moments='mv')
        if self.anneal_method == 'variance':
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
            return {"vs": self.vs, "gs": self.gs, "ns": self.ns}
        if self.critic=="states":
            return {"vs": self.vs_states, "gs": self.gs, "ns": self.ns}
        if self.critic=="state-actions":
            return {"vs": self.vs_stateactions, "gs": self.gs, "ns": self.ns}
        else:
            raise KeyError('critic type not included\nPlease choose from actions, states, state-actions\n')   

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