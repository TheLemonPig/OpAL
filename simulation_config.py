from environments.environment_config import *
from models.model_config import *

environments = [
    bandit_task_,
    # grid_world_
]

models = [
    actor_critic_,
    # opal_,
    opal_star_,
    opal_star_qs_,
    opal_star_var_,
    # q_learning_
]

config_ = {
    "verbose": True,
    "plot": True,
    "verbose_params": {
        'success_metrics': {
            'average': True
        }
    },
    "plot_params": {
        'state_heatmap': {
            'average': True
        },
        'learning_rates': {
            'average': True
        },
        'trends': {
            'cumulative': True,
            'rolling': True
        }
    },
    "epochs": 10000,
    "n_reps": 10,
    "environment_params": environments,
    "model_params": models,
    "seed": range(10)
}
