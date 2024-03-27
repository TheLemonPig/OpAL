from environments.environment_config import *
from models.model_config import *

environments = [
    #bandit_task_,
    grid_world_
]

models = [
    actor_critic_,
    #opal_,
    #opal_star_,
    #opal_star_qs_,
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
            'average': True
        },
        'learning_rates': {
            'average': True
        },
        'trends': {
            'cumulative': False,
            'rolling': True,
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
    "grid_params": {
        'alpha': (0.1, 1.0, 0.1),
        'beta': (1.0, 2.0, 1.0),
        'gamma': (0.9, 1.0, 0.1)},
    "compare": False,
    "compare_param": ('beta', 0.1, 0.5, 0.1)
}
