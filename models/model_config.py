# Model Configs

actor_critic_ = {
    "name": 'ActorCritic',
    "model": 'ActorCritic',
    "alpha": 0.7,
    "beta": 1.0,
    "gamma": 0.0,
}

q_learning_ = {
    "name": 'QLearning',
    "model": 'QLearning',
    "alpha": 0.7,
    "beta": 20.0,
    "gamma": 0.97,
}

opal_star_ = {
    "name": 'OpALStar',
    "model": 'OpALStar',
    "alpha_g": 0.1, # 0.2,
    "alpha_n": 0.1, # 0.2,
    "alpha_c": 0.5, # 0.6,
    "beta": 2.0, # 3.0,
    "gamma": 0.0, # 0.98,
    "rho": 0,
    "phi": 1.0,
    "k": 20.0,
    "T": 100.0,
    "r_mag": 1.0,
    "l_mag": 0.0,
    "anneal_method": 'variance'
}

opal_plus_ = {
    "name": 'OpALPlus',
    "model": 'OpALPlus',
    "alpha_g": 0.1, # 0.2,
    "alpha_n": 0.1, # 0.2,
    "alpha_c": 0.5, # 0.6,
    "beta": 2.0,
    "gamma": 0.0, # 0.98,
    "rho": 0,
    "T": 100.0,
    "r_mag": 1.0,
    "l_mag": 0.0,
    "anneal_method": 'variance'
}
