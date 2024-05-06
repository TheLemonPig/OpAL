from environments.environment_config import *
from models.model_config import *

environments = [
    bandit_task_large_sparse,
    bandit_task_small_rich
    # grid_world_small_sparse,
    # grid_world_small_rich
]

models = [
    # actor_critic_,
    opal_plus_,
    opal_star_,
    # opal_plus_,
    # q_learning_
]

config_ = {
    "verbose": False,
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
        'action_heatmap': {
            'average': True
        },
        'weight_heatmap': {
            'average': True,
            'timesteps': [100,999]
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
    "epochs": 1000,
    "n_reps": 1000,
    "environment_params": environments,
    "model_params": models,
    "seed": range(1000),
    "hyperparams": {
        # 'T': (250000, 750000, 50000),
        'alpha': (0.05,1.01,0.05),
        'alpha_c': [0.025, 0.05, 0.1],
        'beta': (1.0, 10.01, 0.5),
        # 'gamma': (0.95,0.98,0.01),
        #'phi': (0.5,2.5,0.2),
        #'k': (2.0, 22.0, 2.0)
    }
}
