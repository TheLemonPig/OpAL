import numpy as np
from itertools import product
from utils import location_counter, state_action_counter


def success_metrics(config, results, n_reps, average=True, compare=None, verbose=True, test_ratio=0.1, **kwargs):
    if compare:
        print('compare branch not implemented for success_metrics')
    else:
        success_rates = []
    for env_dic in config['environment_params']:
            if env_dic['model'] == 'GridWorld':
                success_results = []
                for mod_dic in config['model_params']:
                    domain = env_dic['state_space']
                    terminals = env_dic['terminal_states']
                    success_terminals = env_dic['success_terminals']
                    if average:
                        location_counts = np.zeros(domain)
                        for n in range(n_reps):
                            state_list = results[env_dic['name']][mod_dic['name']][n]['new_states']
                            if test_ratio > 1/config['epochs']:
                                n_epochs = config['epochs']
                            state_list = state_list[-int(n_epochs*test_ratio):]
                        location_counts += location_counter(state_list, domain)
                        n_success = 0
                        n_failures = 0
                        for terminal in terminals:
                            if terminal in success_terminals:
                                n_success += location_counts[terminal]
                            else:
                                n_failures += location_counts[terminal]
                        n_attempts = n_success + n_failures + 1
                    if verbose:
                            print(f'{mod_dic["name"]} success rate: {np.round(n_success/n_attempts*100, 2)}%')
                    success_results.append(np.round(n_success/n_attempts*100, 2))
            elif env_dic['model'] == 'BanditTask':
                success_results = []
                for mod_dic in config['model_params']:
                    success_state_actions = env_dic['success_state_actions']
                    state_action_space = np.array(env_dic['ps']).shape
                    if average:
                        state_action_counts = np.zeros(state_action_space)
                        for n in range(n_reps):
                            state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                            action_list = results[env_dic['name']][mod_dic['name']][n]['actions']
                        if test_ratio > 1/config['epochs']:
                            n_epochs = config['epochs']
                            action_list = action_list[-int(n_epochs*test_ratio):]
                            state_action_list = [(state[0], action) for (state, action) in zip(state_list, action_list)]
                            state_action_counts += state_action_counter(state_action_list, state_action_space)
                        n_success = 0
                        n_failures = 0
                        for state_x, state_y, action in product(*[range(i) for i in state_action_space]):
                            if (state_x, state_y, action) in success_state_actions:
                                n_success += state_action_counts[state_x, state_y][action]
                            else:
                                n_failures += state_action_counts[state_x, state_y][action]
                        n_attempts = n_success + n_failures
                        if verbose:
                            print(f'{mod_dic["name"]} success rate: {np.round(n_success / n_attempts * 100, 2)}%')
                        success_results.append(np.round(n_success / n_attempts * 100, 2))
    return success_rates

def auc(config, results, n_reps, average=True, verbose=True, **kwargs):
    aucs = []
    for env_dic in config['environment_params']:
        for mod_dic in config['model_params']:
            p_values = results[env_dic['name']][mod_dic['name']][0]['probabilities']
            p_array = np.zeros_like(p_values)
            for n in range(n_reps):
                p_array += results[env_dic['name']][mod_dic['name']][n]['probabilities']
            p_array = p_array / n_reps
            if env_dic['model'] == 'GridWorld':
                ...
            elif env_dic['model'] == 'BanditTask':
                indices = env_dic['success_actions']
            else:
                raise KeyError
            auc = np.round(p_array[:,indices].sum(), decimals=2)
            print(p_array.shape)
            aucs.append(auc)
    return aucs
