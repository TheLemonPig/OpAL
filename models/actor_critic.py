import numpy as np

from models.model import BaseRL
from utils import safe_softmax


class ActorCritic(BaseRL):

    def __init__(self, action_space, state_space, start_state, alpha, beta, gamma=0, name=None, **kwargs):
        BaseRL.__init__(self, action_space=action_space, state_space=state_space, start_state=start_state, name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # TODO: Implement location counter
        # self.location_counter = np.zeros(state_space)
        self.vs = np.ones(state_space) * 0.5
        self.ps = np.ones(state_space+action_space) * 0.5

    def act(self):
        p_values = safe_softmax(self.ps[self.state] * self.beta)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return action

    def update(self, new_state, action, reward):
        delta = self.update_critic(new_state, reward)
        self.update_actor(action, delta)
        self.state = new_state

    def update_critic(self, new_state, reward):
        delta = reward - self.vs[self.state] + self.vs[new_state] * self.gamma
        self.vs[self.state] += self.alpha * delta
        return delta

    def update_actor(self, action, delta):
        self.ps[self.state][action] += delta

    def get_weights(self):
        return {"ps": self.ps, "vs": self.vs}

    def get_optimal_policy(self):
        return self.ps.argmax(axis=-1)
