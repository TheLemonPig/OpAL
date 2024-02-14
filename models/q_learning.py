import numpy as np

from models.model import BaseRL
from utils import safe_softmax


class QLearning(BaseRL):

    def __init__(self, action_space, state_space, start_state, alpha, beta, gamma=0, name=None, **kwargs):
        BaseRL.__init__(self, action_space=action_space, state_space=state_space, start_state=start_state, name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.qs = np.ones(state_space+action_space) * 0.5

    def act(self):
        p_values = safe_softmax(self.qs[self.state]*self.beta)
        action = np.random.choice(len(p_values), 1, p=p_values).item()
        return action

    # TODO: Put model responsible for restarting task

    def update(self, new_state, action, reward):
        delta = reward - self.qs[self.state][action] + self.qs[new_state].max() * self.gamma
        self.qs[self.state][action] += self.alpha * delta
        self.state = new_state

    def get_weights(self):
        return {"qs": self.qs}

    def get_optimal_policy(self):
        return self.qs.argmax(axis=-1)



