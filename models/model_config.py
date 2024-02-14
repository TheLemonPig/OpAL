# Model Configs

actor_critic_ = {
    "name": 'ActorCritic',
    "alpha": 0.1,
    "beta": 1.0,
    "gamma": 0.9,
}

opal_ = {
    "name": 'OpAL',
    "alpha_g": 0.1,
    "alpha_n": 0.1,
    "alpha_c": 0.1,
    "beta": 1.0,
    "gamma": 0.9,
    "rho": 0
}

opal_star_ = {
    "name": 'OpALStar',
    "alpha_g": 0.1,
    "alpha_n": 0.1,
    "alpha_c": 0.1,
    "beta": 1.0,
    "gamma": 0.9,
    "rho": 0,
    "phi": 1.0,
    "k": 1.0,
    "T": 1.0,
    "R_mag": 1.0,
    "L_mag": -1.0,
    "use_qs": False
}

q_learning_ = {
    "name": 'QLearning',
    "alpha": 0.1,
    "beta": 1.0,
    "gamma": 0.9,
}
