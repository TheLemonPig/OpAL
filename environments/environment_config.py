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
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 3): 0.1, (2, 3): 1.0},
    "deterministic": True,
    "success_terminals": [(2, 3)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": []
}

# grid_world_ = {
#     "name": 'GridWorld',
#     "model": 'GridWorld',
#     "state_space": (5, 8),
#     "start_state": (2, 0),
#     "non_terminal_penalty": -0.04,
#     "terminal_states": {(0, 5): 0.2, (2, 7): 0.1, (4, 5): 0.3},
#     "deterministic": True,
#     "success_terminals": [(4, 5)],
#     "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
#     "obstacles": []
# }

grid_world_room = {
    "name": 'GridWorld',
    "model": 'GridWorld',
    "state_space": (5, 8),
    "start_state": (2, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(2, 7): 1.0, (1, 7): 0.01},
    "deterministic": False,
    "success_terminals": [(2, 7)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": [(0, 3), (1, 3), (3, 3), (4, 3)]
}
