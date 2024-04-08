# Model Configs

actor_critic_ = {
    "name": 'ActorCritic',
    "model": 'ActorCritic',
    "alpha": 0.6,
    "beta": 2.0,
    "gamma": 0.6,
}

opal_ = {
    "name": 'OpAL',
    "model": 'OpAL',
    "alpha_g": 0.4,
    "alpha_n": 0.4,
    "alpha_c": 0.4,
    "beta": 5.0,
    "gamma": 0.7,
    "rho": 0
}

opal_star_ = {
    "name": 'OpALStar',
    "model": 'OpALStar',
    "alpha_g": 0.4,
    "alpha_n": 0.4,
    "alpha_c": 0.4,
    "beta": 4.0,
    "gamma": 0.9,
    "rho": 0,
    "phi": 1.7,
    "k": 5.0,
    "T": 50000.0,
    "R_mag": 1.0,
    "L_mag": -1.0,
    "anneal_method": ''
}

opal_star_var_ = {
    "name": 'OpALStarVariance',
    "model": 'OpALStar',
    "alpha_g": 0.5,
    "alpha_n": 0.5,
    "alpha_c": 0.6,
    "beta": 4.0,
    "gamma": 0.6,
    "rho": 0,
    "phi": 1.7,
    "k": 5.0,
    "T": 50000.0,
    "R_mag": 1.0,
    "L_mag": -1.0,
    "anneal_method": 'variance'
}

q_learning_ = {
    "name": 'QLearning',
    "model": 'QLearning',
    "alpha": 0.1,
    "beta": 6.0,
    "gamma": 0.9,
}

opal_star_qs_ = {
    "name": 'OpALStarQs',
    "model": 'OpALStar',
    "alpha_g": 0.5,
    "alpha_n": 0.5,
    "alpha_c": 0.6,
    "beta": 4.0,
    "gamma": 0.6,
    "rho": 0,
    "phi": 1.0,
    "k": 1.0,
    "T": 500000.0,
    "R_mag": 1.0,
    "L_mag": -1.0,
    "anneal_method": 'qs'
}
