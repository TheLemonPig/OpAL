import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from utils import location_counter, state_action_counter


def state_heatmap(simulator, results, n_reps, average=True, compare=False, **kwargs):
    if compare:
        print('compare branch not implemented for success_metrics')
    else:
        for env_dic in simulator.environments:
            if env_dic['name'] == 'GridWorld':
                for mod_dic in simulator.models:
                    domain = env_dic['state_space']
                    if average:
                        location_counts = np.zeros(domain)
                        for n in range(n_reps):
                            state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                            n_attempts = sum(results[env_dic['name']][mod_dic['name']][n]['attempts'])
                            location_counts += location_counter(state_list, domain) / (n_reps * n_attempts)
                        plt.title(f'Average visitations per trial by {mod_dic["name"]}')
                        plt.imshow(location_counts, cmap='viridis', interpolation='nearest')
                        for i in range(location_counts.shape[0]):
                            for j in range(location_counts.shape[1]):
                                plt.text(j, i, f'{location_counts[i, j]:.2f}', ha='center', va='center', color='w')
                        plt.colorbar()
                        plt.show()
                    else:
                        x, y = domain
                        location_counts = np.zeros((x, y, n_reps))
                        for n in range(n_reps):
                            state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                            location_counts[n] += location_counter(state_list, domain)
                        n_rows = int(np.sqrt(n_reps))
                        n_cols = int(np.ceil(n_reps / n_rows))
                        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 8))
                        for n in range(n_reps):
                            row = n // n_rows
                            col = n % n_cols
                            axs[row, col].imshow(location_counts[..., n], cmap='viridis', interpolation='nearest')
                            axs[row, col].set_title(f'Rep {n}')
                            axs[row, col].legend()

                        # Adjust layout to prevent subplot titles from overlapping
                        plt.tight_layout()

                        # Show the plot
                        plt.show()
                        # plt.imshow(location_counts, cmap='viridis', interpolation='nearest')
                        # plt.colorbar()
                        # plt.show()
            elif env_dic['name'] == 'BanditTask':
                for mod_dic in simulator.models:
                    if average:
                        action_space = (len(env_dic['interactions']),)
                        if average:
                            action_counts = np.zeros(action_space)
                            for n in range(n_reps):
                                action_list = results[env_dic['name']][mod_dic['name']][n]['actions']
                                action_counts += action_counter(action_list, action_space)
                            average_counts = (action_counts / action_counts.sum()).reshape((1, -1))
                            plt.title(f'Average actions per trial by {mod_dic["name"]}')
                            plt.imshow(average_counts, cmap='viridis', interpolation='nearest')
                            for j in range(average_counts.shape[1]):
                                plt.text(j, 0, f'{average_counts[0,j]:.2f}', ha='center', va='center', color='w')
                            plt.colorbar()
                            plt.show()


def plot_trends(simulator, results, n_reps, compare=None, **kwargs):
    if compare:
        values = np.arange(*compare[1:])
        if kwargs['probabilities']:
            for env_dic in simulator.environments:
                for mod_dic in simulator.models:
                    for kdx in range(len(results[env_dic['name']][mod_dic['name']])):
                        # p_values = results[env_dic['name']][mod_dic['name']][0]['probabilities']
                        p_array = results[env_dic['name']][mod_dic['name']][kdx]['probabilities']
                        # for n in range(n_reps):
                        #     p_array += results[env_dic['name']][mod_dic['name']][n]['probabilities']
                        # p_array = p_array / n_reps
                        p_indices = list(product(*[range(d) for d in p_array.shape[:-1]]))
                        for jdx, index in enumerate(p_indices):
                            c = ((kdx+1)/len(p_indices), 0, 0, 0.9)
                            plt.plot(p_array[index], color=c)
                        plt.plot(p_array[index], color=c, label=f'{compare[0]}: {np.round(values[kdx],decimals=5)}')
                        if not compare:
                            plt.xlabel('Epochs')
                            plt.ylabel('Probability')
                            plt.legend()
                            plt.title(f"{mod_dic['name']} Probabilities")
                            plt.show()
                if compare:
                    plt.xlabel('Epochs')
                    plt.ylabel('Probability')
                    plt.legend()
                    plt.title(f"{simulator.models[0]['name']} Probabilities")
                    plt.show()
    else:
        if kwargs['cumulative']:
            for env_dic in simulator.environments:
                for mod_dic in simulator.models:
                    avg_cum = np.zeros((len(results[env_dic['name']][mod_dic['name']][0]['cumulative']),))
                    for n in range(n_reps):
                        avg_cum += results[env_dic['name']][mod_dic['name']][n]['cumulative']
                    avg_cum = avg_cum / n_reps
                    plt.plot(np.arange(len(avg_cum)), avg_cum, label=f"{mod_dic['name']} in {env_dic['name']}")
            plt.legend()
            plt.show()
        if kwargs['rolling']:
            roll = 100
            for env_dic in simulator.environments:
                for mod_dic in simulator.models:
                    avg_cum = np.zeros((len(results[env_dic['name']][mod_dic['name']][0]['rolling']),))
                    for n in range(n_reps):
                        avg_cum += results[env_dic['name']][mod_dic['name']][n]['rolling']
                    avg_cum = avg_cum / n_reps
                    plt.plot(np.arange(len(avg_cum[roll:])), avg_cum[roll:], label=f"{mod_dic['name']} in {env_dic['name']}")
            plt.legend()
            plt.show()
        if kwargs['weights']:
            for env_dic in simulator.environments:
                for mod_dic in simulator.models:
                    w_dic = results[env_dic['name']][mod_dic['name']][0]['weights']
                    fig, ax = plt.subplots(1, len(w_dic))
                    for idx, item in enumerate(w_dic.items()):
                        this_ax = ax if len(w_dic) == 1 else ax[idx]
                        k, weights = item
                        w_array = np.zeros_like(weights)
                        for n in range(n_reps):
                            w_array += results[env_dic['name']][mod_dic['name']][n]['weights'][k]
                        w_array = w_array / n_reps
                        w_indices = list(product(*[range(d) for d in w_array.shape[:-1]]))
                        for jdx, index in enumerate(w_indices):
                            this_ax.plot(w_array[index], label=f'{index}', color=((jdx+1)/len(w_indices), 0, 0, 0.5))
                            this_ax.set_xlabel('Epochs')
                            this_ax.set_ylabel('Weight Value')
                            this_ax.legend()
                            this_ax.set_title(f'{k} Weights')
                    fig.suptitle(f"{mod_dic['name']}")
                    plt.show()
        if kwargs['probabilities']:
            for env_dic in simulator.environments:
                for mod_dic in simulator.models:
                    p_values = results[env_dic['name']][mod_dic['name']][0]['probabilities']
                    p_array = np.zeros_like(p_values)
                    for n in range(n_reps):
                        p_array += results[env_dic['name']][mod_dic['name']][n]['probabilities']
                    p_array = p_array / n_reps
                    p_indices = list(product(*[range(d) for d in p_array.shape[:-1]]))
                    for jdx, index in enumerate(p_indices):
                        plt.plot(p_array[index], label=f'{index}', color=((jdx+1)/len(p_indices), 0, 0, 0.5))
                    plt.xlabel('Epochs')
                    plt.ylabel('Probability')
                    plt.legend()
                    plt.title(f"{mod_dic['name']} Probabilities")
                    plt.show()





# def learning_rates(simulator, results, n_reps, average=True, **kwargs):
#     for env_dic in simulator.environments:
#         if env_dic['name'] == 'GridWorld':
#             for mod_dic in simulator.models:
#                 if average:



if __name__ == "__main__":
    ...
