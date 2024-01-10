import numpy as np

from models.model import BaseRL
from utils import tempered_softmax


class ActorCritic(BaseRL):

    def __init__(self, domain_shape, state):
        BaseRL.__init__(self, domain=domain_shape, state=state)

    def act(self):
        ...

    def update(self, new_state, action, reward):
        ...

    def critic(self):
        ...
