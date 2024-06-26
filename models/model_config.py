# Model Configs

actor_critic_ = {
    "name": 'ActorCritic',
    "model": 'ActorCritic',
    "alpha": 0.5,
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

opal_plus_ = {
    "name": 'OpALPlus',
    "model": 'OpALPlus',
    "alpha_g": 0.9,
    "alpha_n": 0.9,
    "alpha_c": 0.05,
    "beta": 1.5,
    "gamma": 0.0,
    "rho": 0,
    "phi": 1.0,
    "k": 20,
    "T": 100.0,
    "r_mag": 1.0,
    "l_mag": 0,
    "anneal_method": 'variance'
}

opal_star_ = {
    "name": 'OpALStar',
    "model": 'OpALStar',
    "alpha_g": 0.5,
    "alpha_n": 0.5,
    "alpha_c": 0.05,
    "beta": 1.0,
    "gamma": 0.0,
    "gamma_h": 0,
    "rho": 0,
    "phi": 1.0,
    "k": 20,
    "T": 100.0,
    "r_mag": 1.0,
    "l_mag": 0,
    "anneal_method": 'variance',
    "hs": False
}

opal_star_hs = {
    "name": 'OpALStarHs',
    "model": 'OpALStar',
    "alpha_g": 1.0,
    "alpha_n": 1.0,
    "alpha_c": 0.05,
    "beta": 4.5,
    "gamma": 0.0,
    "gamma_h": 0.97,
    "rho": 0,
    "phi": 1.5,
    "k": 10,
    "T": 10000.0,
    "r_mag": 1.0,
    "l_mag": 0,
    "anneal_method": 'variance',
    "hs": True
}

opal_star_qs_ = {
    "name": 'OpAL*Qs',
    "model": 'OpALStar',
    "alpha_g": 0.95,
    "alpha_n": 0.95,
    "alpha_c": 0.1,
    "beta": 1.0,
    "gamma": 0.0,
    "rho": 0,
    "phi": 1.0,
    "k": 20,
    "T": 100.0,
    "r_mag": 1.0,
    "l_mag": 0,
    "anneal_method": 'qs'
}
