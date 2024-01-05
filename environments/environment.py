
class BaseEnvironment:

    def __init__(self):
        ...

    def sample(self, choice) -> float:
        ...

    def get_domain(self):
        ...
