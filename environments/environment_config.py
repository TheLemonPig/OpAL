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
grid_world_small_sparse = {
    "name": 'GridWorldSmallSparse',
    "model": 'GridWorld',
    "state_space": (3, 4),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 3): 0.2, (2, 3): 0.3},
    "deterministic": False,
    "success_terminals": [(2, 3)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_world_small_rich = {
    "name": 'GridWorldSmallRich',
    "model": 'GridWorld',
    "state_space": (3, 4),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 3): 0.7, (2, 3): 0.8},
    "deterministic": False,
    "success_terminals": [(2, 3)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_world_large_sparse = {
    "name": 'GridWorldLargeSparse',
    "model": 'GridWorld',
    "state_space": (5, 8),
    "start_state": (2, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 5): 0.2, (2, 7): 0.1, (4, 5): 0.3},
    "deterministic": False,
    "success_terminals": [(4, 5)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": [(2,2),(2,3)]
}
grid_world_large_rich = {
    "name": 'GridWorldLargeRich',
    "model": 'GridWorld',
    "state_space": (5, 8),
    "start_state": (2, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 5): 0.7, (2, 7): 0.6, (4, 5): 0.8},
    "deterministic": False,
    "success_terminals": [(4, 5)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": [(2,2),(2,3)]
}
