import numpy as np

from utils import location_counter, action_counter


def success_metrics(config, results, n_reps, average=True, verbose=True, test_ratio=0.1, **kwargs):
    success_rates = []
    for env_dic in config['environment_params']:
        if env_dic['model'] == 'GridWorld':
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
                    success_rate = np.round(n_success/n_attempts*100, 2)
                    success_rates.append(success_rate)
        elif env_dic['model'] == 'BanditTask':
            for mod_dic in config['model_params']:
                success_actions = env_dic['success_actions']
                action_space = (len(env_dic['interactions']),)
                (n_actions, ) = action_space
                if average:
                    action_counts = np.zeros(action_space)
                    for n in range(n_reps):
                        action_list = results[env_dic['name']][mod_dic['name']][n]['actions']
                        if test_ratio > 1/config['epochs']:
                            n_epochs = config['epochs']
                            action_list = action_list[-int(n_epochs*test_ratio):]
                        action_counts += action_counter(action_list, action_space)
                    n_success = 0
                    n_failures = 0
                    for action in range(n_actions):
                        if action in success_actions:
                            n_success += action_counts[action]
                        else:
                            n_failures += action_counts[action]
                    n_attempts = n_success + n_failures
                    if verbose:
                        print(f'{mod_dic["name"]} success rate: {np.round(n_success / n_attempts * 100, 2)}%')
                    success_rate = np.round(n_success / n_attempts * 100, 2)
                    success_rates.append(success_rate)
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
