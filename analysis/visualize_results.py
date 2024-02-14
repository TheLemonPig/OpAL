import matplotlib.pyplot as plt
import numpy as np

from utils import location_counter, action_counter


def state_heatmap(simulator, results, n_reps, average=True, **kwargs):
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
        elif env_dic['name'] == 'BanditTask':
            for mod_dic in simulator.models:
                if average:
                    action_space = (len(env_dic['interactions']),)
                    if average:
                        action_counts = np.zeros(action_space)
                        for n in range(n_reps):
                            action_list = results[env_dic['name']][mod_dic['name']][n]['actions']
                            action_counts += action_counter(action_list, action_space)
                        average_counts = (action_counts / action_counts.sum()).reshape((1,-1))
                        plt.title(f'Average actions per trial by {mod_dic["name"]}')
                        plt.imshow(average_counts, cmap='viridis', interpolation='nearest')
                        plt.colorbar()
                        plt.show()




if __name__ == "__main__":
    example()
