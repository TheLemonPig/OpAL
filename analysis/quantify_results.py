import numpy as np
from itertools import product
from utils import location_counter, state_action_counter


def success_metrics(simulator, results, n_reps, average=True, compare=None, **kwargs):
    if compare:
        print('compare branch not implemented for success_metrics')
    else:
        for env_dic in simulator.environments:
            if env_dic['name'] == 'GridWorld':
                success_results = []
                for mod_dic in simulator.models:
                    domain = env_dic['state_space']
                    terminals = env_dic['terminal_states']
                    success_terminals = env_dic['success_terminals']
                    if average:
                        location_counts = np.zeros(domain)
                        for n in range(n_reps):
                            state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                            location_counts += location_counter(state_list, domain)
                        n_success = 0
                        n_failures = 0
                        for terminal in terminals:
                            if terminal in success_terminals:
                                n_success += location_counts[terminal]
                            else:
                                n_failures += location_counts[terminal]
                        n_attempts = n_success + n_failures
                        print(f'{mod_dic["name"]} success rate: {np.round(n_success/n_attempts*100, 2)}%')
                        success_results.append(np.round(n_success/n_attempts*100, 2))
                return success_results
            elif env_dic['name'] == 'BanditTask':
                success_results = []
                for mod_dic in simulator.models:
                    success_state_actions = env_dic['success_state_actions']
                    state_action_space = np.array(env_dic['ps']).shape
                    if average:
                        state_action_counts = np.zeros(state_action_space)
                        for n in range(n_reps):
                            state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                            action_list = results[env_dic['name']][mod_dic['name']][n]['actions']
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
                        print(f'{mod_dic["name"]} success rate: {np.round(n_success / n_attempts * 100, 2)}%')
                        success_results.append(np.round(n_success / n_attempts * 100, 2))
                return success_results


