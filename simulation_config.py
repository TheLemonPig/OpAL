from environments.environment_config import *
from models.model_config import *

environments = [
    # bandit_task_,
    grid_world_large_sparse
]

models = [
    # actor_critic_,
    opal_plus_,
    # opal_star_,
    # opal_star_qs_,
    # opal_star_var_,
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
        'action_heatmap': {
            'average': True
        },
        'learning_rates': {
            'average': True
        },
        'trends': {
            'cumulative': True,
            'rolling': True,
            'rho': True
        }
    },
    "epochs": 6000,
    "n_reps": 3,
    "environment_params": environments,
    "model_params": models,
    "seed": range(200)
}
