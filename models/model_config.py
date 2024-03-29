# Model Configs

actor_critic_ = {
    "name": 'ActorCritic',
    "model": 'ActorCritic',
    "alpha": 0.1,
    "beta": 1.0,
    "gamma": 0.0,
}

opal_ = {
    "name": 'OpAL',
    "model": 'OpAL',
    "alpha_g": 0.6,
    "alpha_n": 0.7,
    "alpha_c": 0.7,
    "beta": 1.0,
    "gamma": 0.0,
    "rho": 0
}

opal_star_ = {
    "name": 'OpALStar',
    "model": 'OpALStar',
    "alpha_g": 0.7,
    "alpha_n": 0.7,
    "alpha_c": 0.6,
    "beta": 1.0,
    "gamma": 0.0,
    "rho": 0,
    "phi": 1.5,
    "k": 20.0,
    "T": 500000.0,
    "R_mag": 1.0,
    "L_mag": -1.0,
    "anneal_method": ''
}

opal_star_qs_ = {
    "name": 'OpALStarQs',
    "model": 'OpALStar',
    "alpha_g": 0.7,
    "alpha_n": 0.7,
    "alpha_c": 0.6,
    "beta": 1.0,
    "gamma": 0.0,
    "rho": 0,
    "phi": 1.5,
    "k": 20.0,
    "T": 500000.0,
    "R_mag": 1.0,
    "L_mag": -1.0,
    "anneal_method": 'qs'
}

opal_star_var_ = {
    "name": 'OpALStarVariance',
    "model": 'OpALStar',
    "alpha_g": 0.7,
    "alpha_n": 0.7,
    "alpha_c": 0.6,
    "beta": 1.0,
    "gamma": 0.0,
    "rho": 0,
    "phi": 1.5,
    "k": 20.0,
    "T": 500000.0,
    "R_mag": 1.0,
    "L_mag": -1.0,
    "anneal_method": 'variance'
}

q_learning_ = {
    "name": 'QLearning',
    "model": 'QLearning',
    "alpha": 0.1,
    "beta": 1.0,
    "gamma": 0.0,
}
