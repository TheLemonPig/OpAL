from environments.environment_config import *
from models.model_config import *

environments = [
    # bandit_task_,
    grid_world_small_sparse,
    grid_world_small_rich,
    grid_world_large_sparse,
    grid_world_large_rich
]

models = [
    actor_critic_,
    # opal_plus_,
    # opal_star_,
    q_learning_
]

config_ = {
    "verbose": True,
    "plot": False,
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
    "epochs": 5000,
    "n_reps": 1,
    "environment_params": environments,
    "model_params": models,
    "seed": range(1),
    "hyperparams": {
        'alpha': (0.1, 1.0, 0.1), 
        # 'alpha_c': (0.1, 0.8, 0.1),
        'beta': (2.0, 30.0, 2.0),
        'gamma': (0.95, 0.99, 0.01)
        }
}
