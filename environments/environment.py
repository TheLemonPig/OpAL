
class BaseEnvironment:

    def __init__(self, interactions, state_space, start_state, name):
        self.name = "NamelessEnvironment" if name is None else name
        self.interactions = [(0,)] if interactions is None else interactions
        self.interaction_space = (len(self.interactions),)
        self.state_space = state_space
        self.model_start = (0,) if start_state is None else start_state
        self.model_state = self.model_start

    def interact(self, action) -> float:
        ...

    def restart(self):
        self.model_state = self.model_start

    def at_terminal(self):
        return False
