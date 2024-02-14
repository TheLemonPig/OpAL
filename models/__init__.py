from .actor_critic import ActorCritic
from .OpAL import OpAL
from .OpALStar import OpALStar
from .q_learning import QLearning


model_library = {
    ActorCritic.__name__: ActorCritic,
    OpAL.__name__: OpAL,
    OpALStar.__name__: OpALStar,
    QLearning.__name__: QLearning
}
