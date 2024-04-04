# Environment Configs

bandit_task_ = {
    "name": 'BanditTask',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.2, 0.3, 0.1, 0.2],
    "success_actions": [1],
    "interactions": [0, 1]
}

grid_world_ = {
    "name": 'GridWorld',
    "model": 'GridWorld',
    "state_space": (3, 4),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.0,
    "terminal_states": {(2, 3): 0.8, (0, 3): 0.2},
    "deterministic": True,
    "success_terminals": [(2, 3)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)]
}
