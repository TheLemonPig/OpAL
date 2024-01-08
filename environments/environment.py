
class BaseEnvironment:

    def __init__(self):
        ...

    def interact(self, state, action) -> float:
        ...

    def get_domain(self):
        ...
