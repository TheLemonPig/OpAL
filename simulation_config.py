from environments.environment_config import bandit_task_, grid_world_
from models.model_config import actor_critic_, opal_, opal_star_, q_learning_

environments = [
    bandit_task_,
    # grid_world_
]

models = [
    # actor_critic_,
    # opal_,
    opal_star_,
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
        }
    },
    "epochs": 10000,
    "n_reps": 1,
    "environment_params": environments,
    "model_params": models,
}
