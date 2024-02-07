from .actor_critic import ActorCritic
from .OpAL import OpAL
from .OpALStar import OpALStar
from .q_learning import QLearning

model_library = {"ActorCritic": ActorCritic, "OpAL": OpAL, "OpAlStar": OpALStar, "QLearning": QLearning}
