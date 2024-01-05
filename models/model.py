

# Basic Template for all RL models
class BaseRL:

    def __init__(self, domain, state):
        self.state = state
        self.domain = domain

    def act(self):
        ...

    def update(self, *args):
        ...

    def get_predictions(self):
        ...
