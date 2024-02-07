import matplotlib.pyplot as plt
import numpy as np

from utils import location_counter


def success_metrics(results, n_reps, **kwargs):
    for simulator_dic in results:
        domain = simulator_dic['simulator'].model.domain
        terminals = simulator_dic['simulator'].environment.terminals
        success_terminals = simulator_dic['simulator'].environment.config['success_terminals']
        if kwargs['average']:
            location_counts = np.zeros(domain)
            for n in range(n_reps):
                state_list = simulator_dic['states'][n]
                location_counts += location_counter(state_list, domain) / n_reps
            n_success = 0
            n_failures = 0
            for success_terminal in success_terminals:
                n_success += location_counts[success_terminal]
            for terminal in terminals:
                if terminal not in success_terminals:
                    n_failures += location_counts[terminal]
            n_attempts = n_success + n_failures
            print(f'Success rate: {np.round(n_success/n_attempts*100, 2)}%')


