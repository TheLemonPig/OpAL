
class BaseEnvironment:

    def __init__(self, name=None, **config):
        self.name = name
        self.config = config

    def interact(self, state, action) -> float:
        ...

    def get_domain(self):
        ...
