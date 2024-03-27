import numpy as np

from environments.environment import BaseEnvironment


class BanditTask(BaseEnvironment):

    def __init__(self, interactions, state_space, start_state, rewards=None, deterministic=False, ps=None, name=None,
                 **kwargs):
        BaseEnvironment.__init__(self, interactions=interactions, state_space=state_space, start_state=start_state,
                                 name=name)
        self.deterministic = deterministic
        self.state_action_space = np.array(ps)
        if len(state_space) == 1 and state_space[0] == 1:
            self.state_action_space = np.array(ps).reshape((1, -1))
        self.rewards = np.ones_like(self.state_action_space) if rewards is None else np.array(rewards)

    def interact(self, action):
        p = self.state_action_space[self.model_state][action]
        if self.deterministic:
            reward = p
        else:
            reward = np.random.choice(2, 1, p=[1-p, p]).item()
            self.model_state = tuple(np.random.randint(self.state_space))
        reward = reward * self.rewards[self.model_state][action]
        return self.model_state, reward
