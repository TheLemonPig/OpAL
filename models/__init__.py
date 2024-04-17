from .actor_critic import ActorCritic
from .OpALPlus import OpALPlus
from .OpALStar import OpALStar
from .q_learning import QLearning

model_library = {
    ActorCritic.__name__: ActorCritic,
    OpALPlus.__name__: OpALPlus,
    OpALStar.__name__: OpALStar,
    QLearning.__name__: QLearning
}
