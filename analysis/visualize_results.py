import matplotlib.pyplot as plt
import numpy as np

from utils import location_counter, action_counter


def state_heatmap(simulator, results, n_reps, average=True, **kwargs):
    for env_dic in simulator.environments:
        if env_dic['model'] == 'GridWorld':
            for mod_dic in simulator.models:
                domain = env_dic['state_space']
                if average:
                    location_counts = np.zeros(domain)
                    for n in range(n_reps):
                        state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                        n_attempts = sum(results[env_dic['name']][mod_dic['name']][n]['attempts'])
                        location_counts += location_counter(state_list, domain) / (n_reps * n_attempts)
                    plt.title(f'Average visitations per trial by {mod_dic["name"]} in {env_dic["name"]}')
                    for i in range(location_counts.shape[0]):
                        for j in range(location_counts.shape[1]):
                            if env_dic['obstacles'] is None or (i, j) not in env_dic['obstacles']:
                                plt.text(j, i, f'{location_counts[i, j]:.2f}', ha='center', va='center', color='w')
                    for loc in env_dic['terminal_states']:
                        if loc in env_dic['success_terminals']:
                            circle = plt.Circle((loc[1], loc[0]), 0.5, color='green', fill=False)
                        else:
                            circle = plt.Circle((loc[1], loc[0]), 0.5, color='red', fill=False)
                        plt.gca().add_patch(circle)
                    if env_dic['obstacles'] is not None:
                        for loc in env_dic['obstacles']:
                            square = plt.Rectangle((loc[1]-0.5, loc[0]-0.5), 1.0, 1.0, color='gray', fill=True)
                            plt.gca().add_patch(square)
                    plt.imshow(location_counts, cmap='viridis', interpolation='nearest')
                    plt.colorbar()
                    plt.show()
                else:
                    x, y = domain
                    location_counts = np.zeros((x, y, n_reps))
                    for n in range(n_reps):
                        state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                        location_counts[n] = location_counter(state_list, domain)
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
        elif env_dic['model'] == 'BanditTask':
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


def plot_trends(simulator, results, n_reps, **kwargs):
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
            plt.title('Rolling Average Reward')
            plt.show()
    if kwargs['rho']:
        for env_dic in simulator.environments:
            avg_cum = None
            for mod_dic in simulator.models:
                if 'rho' in mod_dic.keys():
                    avg_cum = np.zeros((len(results[env_dic['name']][mod_dic['name']][0]['rolling']),))
                    for n in range(n_reps):
                        avg_cum += results[env_dic['name']][mod_dic['name']][n]['rho']
                    avg_cum = avg_cum / n_reps
                    plt.xlabel('epochs')
                    plt.ylabel('rho')
                    plt.plot(np.arange(len(avg_cum)), avg_cum, label=f"{mod_dic['name']} in {env_dic['name']}")
            if avg_cum is not None:
                plt.legend()
                plt.title('Rho Value over Training')
                plt.show()



# def learning_rates(simulator, results, n_reps, average=True, **kwargs):
#     for env_dic in simulator.environments:
#         if env_dic['name'] == 'GridWorld':
#             for mod_dic in simulator.models:
#                 if average:



if __name__ == "__main__":
    ...
