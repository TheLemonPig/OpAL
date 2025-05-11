# Environment Configs

bandit_task_large_sparse = {
    "name": 'BanditTaskLargeSparse',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.3, 0.2, 0.2, 0.2, 0.2, 0.2],
    "success_actions": [0],
    "interactions": [0, 1, 2, 3, 4, 5]
}

bandit_task_small_sparse = {
    "name": 'BanditTaskSmallSparse',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.3, 0.2],
    "success_actions": [0],
    "interactions": [0, 1]
}

bandit_task_small_rich = {
    "name": 'BanditTaskSmallRich',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.8, 0.7],
    "success_actions": [0],
    "interactions": [0, 1]
}
bandit_task_medium_sparse = {
    "name": 'BanditTaskMediumSparse',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.3, 0.2, 0.2, 0.2],
    "success_actions": [0],
    "interactions": [0, 1, 2, 3]
}

bandit_task_medium_rich = {
    "name": 'BanditTaskMediumRich',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.8, 0.7, 0.7, 0.7],
    "success_actions": [0],
    "interactions": [0, 1, 2, 3]
}

bandit_task_small_binary_gaussian = {
    "name": 'BanditTaskSmallBinaryGaussian',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [1.0, 0.0],
    "success_actions": [0],
    "interactions": [0, 1],
    "stds": [0.5, 0.5]
}

bandit_task_small_nonbinary_gaussian = {
    "name": 'BanditTaskSmallNonbinaryGaussian',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [2.5, -1.5],
    "success_actions": [0],
    "interactions": [0, 1],
    "stds": [2.0, 2.0]
}

bandit_task_bogacz_gaussian = {
    "name": 'BanditTaskBogaczGaussian',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "guassian": True,
    "ps": [0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
    "success_actions": [0],
    "interactions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "stds": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
}

bandit_task_bogacz_sparse_bernoulli = {
    "name": 'BanditTaskBogaczSparseBernoulli',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "guassian": False,
    "ps": [0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
    "success_actions": [0],
    "interactions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}

bandit_task_bogacz_rich_bernoulli = {
    "name": 'BanditTaskBogaczRichBernoulli',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "guassian": False,
    "ps": [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    "success_actions": [0],
    "interactions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}

bandit_task_mini_bogacz_gaussian = {
    "name": 'BanditTaskMiniBogaczGaussian',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "guassian": True,
    "ps": [0.55, 0.45],
    "success_actions": [0],
    "interactions": [0, 1],
    "stds": [0.3, 0.3]
}

bandit_task_mini_bogacz_sparse_bernoulli = {
    "name": 'BanditTaskMiniBogaczSparseBernoulli',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "guassian": False,
    "ps": [0.55, 0.45],
    "success_actions": [0],
    "interactions": [0, 1],
}

bandit_task_mini_bogacz_rich_bernoulli = {
    "name": 'BanditTaskMiniBogaczRichBernoulli',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "guassian": False,
    "ps": [0.9, 0.8],
    "success_actions": [0],
    "interactions": [0, 1],
}

# In-between worlds
grid_bandit_small_sparse = {
    "name": 'GridBanditSmallSparse',
    "model": 'GridWorld',
    "state_space": (3, 1),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 0): 0.2, (2, 0): 0.3},
    "deterministic": False,
    "success_terminals": [(2, 0)],
    "interactions": [(1, 0), (-1, 0)],
    "obstacles": None
}
grid_bandit_small_rich = {
    "name": 'GridBanditSmallRich',
    "model": 'GridWorld',
    "state_space": (3, 1),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 0): 0.7, (2, 0): 0.8},
    "deterministic": False,
    "success_terminals": [(2, 0)],
    "interactions": [(1, 0), (-1, 0)],
    "obstacles": None
}
grid_bandit_medium_sparse = {
    "name": 'GridBanditMediumSparse',
    "model": 'GridWorld',
    "state_space": (3, 3),
    "start_state": (1, 1),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(1, 0): 0.2, (0, 1): 0.2, (1, 2): 0.2, (2, 1): 0.3},
    "deterministic": False,
    "success_terminals": [(2, 1)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_bandit_medium_rich = {
    "name": 'GridBanditMediumRich',
    "model": 'GridWorld',
    "state_space": (3, 3),
    "start_state": (1, 1),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(1, 0): 0.7, (0, 1): 0.7, (1, 2): 0.7, (2, 1): 0.8},
    "deterministic": False,
    "success_terminals": [(2, 1)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}

# Grid Worlds
grid_world_3x1_sparse = {
    "name": 'GridWorld3x1Sparse',
    "model": 'GridWorld',
    "state_space": (3, 1),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 0): 0.2, (2, 0): 0.3},
    "deterministic": False,
    "success_terminals": [(2, 0)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_world_3x1_rich = {
    "name": 'GridWorld3x1Rich',
    "model": 'GridWorld',
    "state_space": (3, 1),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 0): 0.7, (2, 0): 0.8},
    "deterministic": False,
    "success_terminals": [(2, 0)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_world_3x2_sparse = {
    "name": 'GridWorld3x2Sparse',
    "model": 'GridWorld',
    "state_space": (3, 2),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 1): 0.2, (2, 1): 0.3},
    "deterministic": False,
    "success_terminals": [(2, 1)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_world_3x2_rich = {
    "name": 'GridWorld3x2Rich',
    "model": 'GridWorld',
    "state_space": (3, 2),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 1): 0.7, (2, 1): 0.8},
    "deterministic": False,
    "success_terminals": [(2, 1)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_world_3x3_sparse = {
    "name": 'GridWorld3x3Sparse',
    "model": 'GridWorld',
    "state_space": (3, 3),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 2): 0.2, (2, 2): 0.3},
    "deterministic": False,
    "success_terminals": [(2, 2)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
}
grid_world_3x3_rich = {
    "name": 'GridWorld3x3Rich',
    "model": 'GridWorld',
    "state_space": (3, 3),
    "start_state": (1, 0),
    "non_terminal_penalty": -0.04,
    "terminal_states": {(0, 2): 0.7, (2, 2): 0.8},
    "deterministic": False,
    "success_terminals": [(2, 2)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": None
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
    "terminal_states": {(0, 5): 0.7, (2, 7): 0.6, (4, 6): 0.8},
    "deterministic": False,
    "success_terminals": [(4, 6)],
    "interactions": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "obstacles": [(0,0),(1,3),(2,3),(4,3),(1,0),(1,1),(3,0),(3,1)]
}

bandit_task_pigeon_easy = {
    "name": 'BanditTaskPigeonEasy',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.05, 0.4, 0.75],
    "success_actions": [2],
    "interactions": [0, 1, 2]
}
bandit_task_pigeon_normal = {
    "name": 'BanditTaskPigeonNormal',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.15, 0.4, 0.65],
    "success_actions": [2],
    "interactions": [0, 1, 2]
}
bandit_task_pigeon_difficult = {
    "name": 'BanditTaskPigeonDifficult',
    "model": 'BanditTask',
    "state_space": (1,),
    "start_state": (0,),
    "deterministic": False,
    "ps": [0.25, 0.4, 0.55],
    "success_actions": [2],
    "interactions": [0, 1, 2]
}