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
    "state_space": (5, 8),
    "start_state": (2, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 5): 0.2, (2, 7): 0.1, (4, 5): 0.3},
    "deterministic": False,
    "success_terminals": [(4, 5)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)]
}
