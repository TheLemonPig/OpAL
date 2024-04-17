# Model Configs

actor_critic_ = {
    "name": 'ActorCritic',
    "model": 'ActorCritic',
    "alpha": 0.7,
    "beta": 1.0,
    "gamma": 0.98,
}

q_learning_ = {
    "name": 'QLearning',
    "model": 'QLearning',
    "alpha": 0.7,
    "beta": 20.0,
    "gamma": 0.97,
}

opal_star_ = {
    "name": 'OpAL*',
    "model": 'OpALStar',
    "alpha_g": 0.3,
    "alpha_n": 0.3,
    "alpha_c": 0.7,
    "beta": 3.0,
    "gamma": 0.97,
    "rho": 0,
    "phi": 1.7,
    "k": 5.0,
    "T": 50000.0,
    "R_mag": 1.0,
    "L_mag": -0.04,
    "anneal_method": 'variance'
}

opal_plus_ = {
    "name": 'OpAL+',
    "model": 'OpALPlus',
    "alpha_g": 0.4,
    "alpha_n": 0.6,
    "alpha_c": 0.6,
    "beta": 2.0,
    "gamma": 0.98,
    "rho": 0,
    "phi": 1.7,
    "k": 5.0,
    "T": 50000.0,
    "R_mag": 1.0,
    "L_mag": -0.04,
    "anneal_method": 'variance'
}
