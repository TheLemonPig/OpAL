# Environment Configs

bandit_task_ = {
    "name": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.3, 0.7],
    "success_actions": [1],
    "interactions": [0, 1]
}

grid_world_ = {
    "name": 'GridWorld',
    "state_space": (3, 4),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.0,
    "terminal_states": {(2, 3): -1, (0, 3): 1},
    "deterministic": True,
    "success_terminals": [(0, 3)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)]
}
