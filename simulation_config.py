# Meta Config
config_ = {
        "epochs": 1000,
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
        "type": "solo",
        "n_reps": 1000
    }

# Environment Configs
grid_world_ = {
    "name": 'GridWorld',
    "domain": (3, 4),
    "start": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminals": {(2, 3): 0.7, (0, 3): 0.8},
    "deterministic": False,
    "success_terminals": [(0, 3)]
}
bandit_ = {
    "name": 'GridWorld',
    "domain": (3, 4),
    "start": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminals": {(2, 3): 0.7, (0, 3): 0.8},
    "deterministic": False,
    "success_terminals": [(0, 3)]
}

environment_params_ = grid_world_

# Simulation Configs
simulator_0_ = {
    "interaction_space": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "epochs": config_['epochs'],
    "max_steps": 100,
    "results": ["states", "rewards"],
    "start": environment_params_['start'],
    "success_terminals": environment_params_['success_terminals'],
    "type": config_['type']
}

simulator_params_ = simulator_0_

# Model Configs
actor_critic_ = {
    "name": 'ActorCritic',
    "alpha": 0.1,
    "beta": 1.0,
    "gamma": 0.9,
    "n_actions": len(simulator_params_['interaction_space']),
    "domain": environment_params_['domain'],
    "state": environment_params_['start']
}

model_params_ = actor_critic_

params_ = [[model_params_], [environment_params_], simulator_params_]
