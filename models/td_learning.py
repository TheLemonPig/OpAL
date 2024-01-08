from environments.environment import BaseEnvironment


# RL occurs both over time and space

class TDLearning(BaseEnvironment):

    def __init__(self):
        BaseEnvironment.__init__(self)
        # state_action_matrix

    def act(self):
        ...

    def update(self, choice, reward):
        ...
