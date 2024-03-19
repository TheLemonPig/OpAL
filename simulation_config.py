from environments.environment_config import *
from models.model_config import *

environments = [
    bandit_task_,
    #grid_world_
]

models = [
    actor_critic_,
    #opal_,
    opal_star_,
    opal_star_qs_,
    opal_star_var_,
    #q_learning_
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
            'average': False
        },
        'learning_rates': {
            'average': False
        },
        'trends': {
            'cumulative': False,
            'rolling': False,
            'weights': True,
            'probabilities': True,
        }
    },
    "epochs": 250,
    "n_reps": 20,
    "environment_params": environments,
    "model_params": models,
    "seed": range(20),
    "grid_search": False,
    "grid_params": {'beta': (0.1, 2.1, 0.1),
                    'gamma': (0.1, 0.2, 0.1)},
    "compare": False,
    "compare_param": ('beta', 0.1, 0.5, 0.1)
}
