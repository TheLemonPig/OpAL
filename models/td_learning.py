from models.q_learning import QLearning


# RL occurs both over time and space

class TDLearning(QLearning):

    def __init__(self, actions, domain_shape, state, lr, temperature):
        QLearning.__init__(self)
        # state_action_matrix

    def act(self):
        ...

    def update(self, choice, reward):
        ...
