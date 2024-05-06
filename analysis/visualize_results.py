import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from itertools import product

from utils import location_counter, action_counter, get_square_triangles, state_action_counter


def state_heatmap(config, results, n_reps, average=True, **kwargs):
    for env_dic in config['environment_params']:
        if env_dic['model'] == 'GridWorld':
            for mod_dic in config['model_params']:
                domain = env_dic['state_space']
                if average:
                    location_counts = np.zeros(domain)
                    for n in range(n_reps):
                        state_list = results[env_dic['name']][mod_dic['name']][n]['new_states']
                        n_attempts = sum(results[env_dic['name']][mod_dic['name']][n]['attempts'])
                        location_counts += location_counter(state_list, domain) / (n_reps * n_attempts)
                    plt.title(f'Average visitations per trial by {mod_dic["name"]} in {env_dic["name"]}')
                    for i in range(location_counts.shape[0]):
                        for j in range(location_counts.shape[1]):
                            if env_dic['obstacles'] is None or (i, j) not in env_dic['obstacles']:
                                plt.text(j, i, f'{np.round(location_counts[i, j],decimals=2)}', ha='center', va='center', color='w')
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
                        state_list = results[env_dic['name']][mod_dic['name']][n]['new_states']
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
            for mod_dic in config['model_params']:
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
                            plt.text(j, 0, f'{np.round(average_counts[0,j],decimals=2)}', ha='center', va='center', color='w')
                        plt.colorbar()
                        plt.show()


def action_heatmap(config, results, n_reps, average=True, **kwargs):
    for env_dic in config['environment_params']:
        if env_dic['model'] == 'GridWorld':
            for mod_dic in config['model_params']:
                fig, ax = plt.subplots()
                domain = env_dic['state_space']
                state_action_space = domain + (len(env_dic['interactions']),)
                sa_counts = np.zeros(state_action_space)
                for n in range(n_reps):
                    state_list = results[env_dic['name']][mod_dic['name']][n]['states']
                    action_list = results[env_dic['name']][mod_dic['name']][n]['actions']
                    state_x = [state[0] for state in state_list]
                    state_y = [state[1] for state in state_list]
                    state_action_list = list(zip(state_x, state_y, action_list))
                    n_attempts = sum(results[env_dic['name']][mod_dic['name']][n]['attempts'])
                    sa_counts += state_action_counter(state_action_list, state_action_space) / (n_reps * n_attempts)
                plt.title(f'Average state-actions per trial by {mod_dic["name"]} in {env_dic["name"]}')
                for i in range(sa_counts.shape[0]):
                    for j in range(sa_counts.shape[1]):
                        square_triangles = get_square_triangles(j, domain[0]-i-1, 1)
                        for k, triangle_coords in enumerate(square_triangles):
                            k_i = (((k+1) % 2)/3) * ((-1) ** (k // 2))
                            k_j = ((k % 2)/3) * ((-1) ** (k // 2))
                            if (domain[0] - i - 1, j) not in env_dic['terminal_states'] and \
                                    (env_dic['obstacles'] is None or (domain[0] - i - 1, j) not in env_dic['obstacles']):
                                plt.text(j+k_j+0.5, domain[0]-(i+k_i+0.5), f'{np.round(sa_counts[i, j, k],decimals=2)}', ha='center', va='center', color='w')
                                triangle = Polygon(triangle_coords, closed=True,
                                                   color=plt.cm.viridis(sa_counts[i, j, k]))
                                ax.add_patch(triangle)
                for loc in env_dic['terminal_states']:
                    square = plt.Rectangle((loc[1], domain[0] - loc[0] - 1), 1.0, 1.0, color='black', fill=True)
                    plt.gca().add_patch(square)
                    if loc in env_dic['success_terminals']:
                        circle = plt.Circle(((loc[1]+0.5), domain[0]-loc[0]-0.5), 0.5, color='green', fill=False)
                    else:
                        circle = plt.Circle((loc[1]+0.5, domain[0]-loc[0]-0.5), 0.5, color='red', fill=False)
                    plt.gca().add_patch(circle)
                    plt.text((loc[1] + 0.5), domain[0] - loc[0] - 0.5,
                             f'{np.round(env_dic["terminal_states"][loc],decimals=2)}', ha='center', va='center', color='w')
                if env_dic['obstacles'] is not None:
                    for loc in env_dic['obstacles']:
                        square = plt.Rectangle((loc[1], domain[0]-loc[0]-1), 1.0, 1.0, color='gray', fill=True)
                        plt.gca().add_patch(square)
                # Adjusting plot
                ax.set_xlim(0, domain[1])
                ax.set_ylim(0, domain[0])
                ax.set_aspect('equal', adjustable='box')
                ax.set_xticks(np.arange(domain[1]))
                ax.set_yticks(np.arange(domain[0]))
                plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Weights')
                plt.grid(True)
                plt.show()


def weight_heatmap(config, results, n_reps, average=True, timesteps=None, **kwargs):
    timesteps = [-1,] if timesteps is None else timesteps
    for env_dic in config['environment_params']:
        if env_dic['model'] == 'GridWorld':
            for mod_dic in config['model_params']:
                for t in timesteps:
                    weights = results[env_dic['name']][mod_dic['name']][0]['weights']
                    for key in weights.keys():
                        array = weights[key][...,t]
                        if len(array.shape) == 1:
                            raise NotImplementedError
                        if len(array.shape) == 2:
                            domain = array.shape
                            value_counts = np.zeros(domain)
                            for n in range(n_reps):
                                value_counts += results[env_dic['name']][mod_dic['name']][n]['weights'][key][...,t] / n_reps
                            plt.title(f'Average {key} weights on timestep {t} by {mod_dic["name"]} in {env_dic["name"]}')
                            for i in range(value_counts.shape[0]):
                                for j in range(value_counts.shape[1]):
                                    if env_dic['obstacles'] is None or (i, j) not in env_dic['obstacles']:
                                        plt.text(j, i, f'{np.round(value_counts[i, j],decimals=2)}', ha='center', va='center', color='w')
                            for loc in env_dic['terminal_states']:
                                if loc in env_dic['success_terminals']:
                                    circle = plt.Circle((loc[1], loc[0]), 0.5, color='green', fill=False)
                                else:
                                    circle = plt.Circle((loc[1], loc[0]), 0.5, color='red', fill=False)
                                plt.gca().add_patch(circle)
                            if env_dic['obstacles'] is not None:
                                for loc in env_dic['obstacles']:
                                    square = plt.Rectangle((loc[1] - 0.5, loc[0] - 0.5), 1.0, 1.0, color='gray', fill=True)
                                    plt.gca().add_patch(square)
                            plt.imshow(value_counts, cmap='viridis', interpolation='nearest')
                            plt.colorbar()
                            plt.show()
                        elif len(array.shape) == 3:
                            fig, ax = plt.subplots()
                            domain = env_dic['state_space']
                            sa_counts = np.zeros(array.shape)
                            for n in range(n_reps):
                                sa_counts += results[env_dic['name']][mod_dic['name']][n]['weights'][key][...,t] / n_reps
                            plt.title(f'Average {key} weights timestep {t} by {mod_dic["name"]} in {env_dic["name"]}')
                            for i in range(sa_counts.shape[0]):
                                for j in range(sa_counts.shape[1]):
                                    square_triangles = get_square_triangles(j, domain[0] - i - 1, 1)
                                    for k, triangle_coords in enumerate(square_triangles):
                                        k_i = (((k + 1) % 2) / 3) * ((-1) ** (k // 2))
                                        k_j = ((k % 2) / 3) * ((-1) ** (k // 2))
                                        if (domain[0] - i - 1, j) not in env_dic['terminal_states'] and \
                                                (env_dic['obstacles'] is None or (domain[0] - i - 1, j) not in env_dic[
                                                    'obstacles']):
                                            plt.text(j + k_j + 0.5, domain[0] - (i + k_i + 0.5),
                                                     f'{np.round(sa_counts[i, j, k],decimals=2)}', ha='center', va='center', color='w')
                                            triangle = Polygon(triangle_coords, closed=True,
                                                               color=plt.cm.viridis(sa_counts[i, j, k]))
                                            ax.add_patch(triangle)
                            for loc in env_dic['terminal_states']:
                                square = plt.Rectangle((loc[1], domain[0] - loc[0] - 1), 1.0, 1.0, color='black', fill=True)
                                plt.gca().add_patch(square)
                                if loc in env_dic['success_terminals']:
                                    circle = plt.Circle(((loc[1] + 0.5), domain[0] - loc[0] - 0.5), 0.5, color='green',
                                                        fill=False)
                                else:
                                    circle = plt.Circle((loc[1] + 0.5, domain[0] - loc[0] - 0.5), 0.5, color='red',
                                                        fill=False)
                                plt.gca().add_patch(circle)
                                plt.text((loc[1] + 0.5), domain[0] - loc[0] - 0.5,
                                         f'{np.round(env_dic["terminal_states"][loc],decimals=2)}', ha='center', va='center', color='w')
                            if env_dic['obstacles'] is not None:
                                for loc in env_dic['obstacles']:
                                    square = plt.Rectangle((loc[1], domain[0] - loc[0] - 1), 1.0, 1.0, color='gray',
                                                           fill=True)
                                    plt.gca().add_patch(square)
                            # Adjusting plot
                            ax.set_xlim(0, domain[1])
                            ax.set_ylim(0, domain[0])
                            ax.set_aspect('equal', adjustable='box')
                            ax.set_xticks(np.arange(domain[1]))
                            ax.set_yticks(np.arange(domain[0]))
                            plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Weights')
                            plt.grid(True)
                            plt.show()
                        else:
                            raise NotImplementedError


def policy_heatmap(config, results, n_reps, average=True, timesteps=None, **kwargs):
    timesteps = [-1,] if timesteps is None else timesteps
    for env_dic in config['environment_params']:
        if env_dic['model'] == 'GridWorld':
            for mod_dic in config['model_params']:
                for t in timesteps:
                    ps = results[env_dic['name']][mod_dic['name']][0]['probabilities']
                    array = ps[...,t]
                    if len(array.shape) == 1:
                        raise NotImplementedError
                    if len(array.shape) == 2:
                        domain = array.shape
                        value_counts = np.zeros(domain)
                        for n in range(n_reps):
                            value_counts += results[env_dic['name']][mod_dic['name']][n]['probabilities'][...,t] / n_reps
                        plt.title(f'Average policy on timestep {t} by {mod_dic["name"]} in {env_dic["name"]}')
                        for i in range(value_counts.shape[0]):
                            for j in range(value_counts.shape[1]):
                                if env_dic['obstacles'] is None or (i, j) not in env_dic['obstacles']:
                                    plt.text(j, i, f'{np.round(value_counts[i, j],decimals=2)}', ha='center', va='center', color='w')
                        for loc in env_dic['terminal_states']:
                            if loc in env_dic['success_terminals']:
                                circle = plt.Circle((loc[1], loc[0]), 0.5, color='green', fill=False)
                            else:
                                circle = plt.Circle((loc[1], loc[0]), 0.5, color='red', fill=False)
                            plt.gca().add_patch(circle)
                        if env_dic['obstacles'] is not None:
                            for loc in env_dic['obstacles']:
                                square = plt.Rectangle((loc[1] - 0.5, loc[0] - 0.5), 1.0, 1.0, color='gray', fill=True)
                                plt.gca().add_patch(square)
                        plt.imshow(value_counts, cmap='viridis', interpolation='nearest')
                        plt.colorbar()
                        plt.show()
                    elif len(array.shape) == 3:
                        fig, ax = plt.subplots()
                        domain = env_dic['state_space']
                        sa_counts = np.zeros(array.shape)
                        for n in range(n_reps):
                            sa_counts += results[env_dic['name']][mod_dic['name']][n]['probabilities'][...,t] / n_reps
                        plt.title(f'Average policy timestep {t} by {mod_dic["name"]} in {env_dic["name"]}')
                        for i in range(sa_counts.shape[0]):
                            for j in range(sa_counts.shape[1]):
                                square_triangles = get_square_triangles(j, domain[0] - i - 1, 1)
                                for k, triangle_coords in enumerate(square_triangles):
                                    k_i = (((k + 1) % 2) / 3) * ((-1) ** (k // 2))
                                    k_j = ((k % 2) / 3) * ((-1) ** (k // 2))
                                    if (domain[0] - i - 1, j) not in env_dic['terminal_states'] and \
                                            (env_dic['obstacles'] is None or (domain[0] - i - 1, j) not in env_dic[
                                                'obstacles']):
                                        plt.text(j + k_j + 0.5, domain[0] - (i + k_i + 0.5),
                                                 f'{np.round(sa_counts[i, j, k],decimals=2)}', ha='center', va='center', color='w')
                                        triangle = Polygon(triangle_coords, closed=True,
                                                           color=plt.cm.viridis(sa_counts[i, j, k]))
                                        ax.add_patch(triangle)
                        for loc in env_dic['terminal_states']:
                            square = plt.Rectangle((loc[1], domain[0] - loc[0] - 1), 1.0, 1.0, color='black', fill=True)
                            plt.gca().add_patch(square)
                            if loc in env_dic['success_terminals']:
                                circle = plt.Circle(((loc[1] + 0.5), domain[0] - loc[0] - 0.5), 0.5, color='green',
                                                    fill=False)
                            else:
                                circle = plt.Circle((loc[1] + 0.5, domain[0] - loc[0] - 0.5), 0.5, color='red',
                                                    fill=False)
                            plt.gca().add_patch(circle)
                            plt.text((loc[1] + 0.5), domain[0] - loc[0] - 0.5,
                                     f'{np.round(env_dic["terminal_states"][loc],decimals=2)}', ha='center', va='center', color='w')
                        if env_dic['obstacles'] is not None:
                            for loc in env_dic['obstacles']:
                                square = plt.Rectangle((loc[1], domain[0] - loc[0] - 1), 1.0, 1.0, color='gray',
                                                       fill=True)
                                plt.gca().add_patch(square)
                        # Adjusting plot
                        ax.set_xlim(0, domain[1])
                        ax.set_ylim(0, domain[0])
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_xticks(np.arange(domain[1]))
                        ax.set_yticks(np.arange(domain[0]))
                        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Policy')
                        plt.grid(True)
                        plt.show()
                    else:
                        raise NotImplementedError


def plot_trends(config, results, n_reps, **kwargs):
    if kwargs['cumulative']:
        for env_dic in config['environment_params']:
            for mod_dic in config['model_params']:
                avg_cum = np.zeros((len(results[env_dic['name']][mod_dic['name']][0]['cumulative']),))
                for n in range(n_reps):
                    avg_cum += results[env_dic['name']][mod_dic['name']][n]['cumulative']
                avg_cum = avg_cum / n_reps
                plt.plot(np.arange(len(avg_cum)), avg_cum, label=f"{mod_dic['name']} in {env_dic['name']}")
            plt.legend()
            plt.show()
    if kwargs['rolling']:
        roll = 100
        for env_dic in config['environment_params']:
            for mod_dic in config['model_params']:
                avg_cum = np.zeros((len(results[env_dic['name']][mod_dic['name']][0]['rolling']),))
                for n in range(n_reps):
                    avg_cum += results[env_dic['name']][mod_dic['name']][n]['rolling']
                avg_cum = avg_cum / n_reps
                plt.plot(np.arange(len(avg_cum[roll:])), avg_cum[roll:], label=f"{mod_dic['name']} in {env_dic['name']}")
            plt.legend()
            plt.title('Rolling Average Reward')
            plt.show()
    if kwargs['rho']:
        for env_dic in config['environment_params']:
            avg_cum = None
            for mod_dic in config['model_params']:
                if 'rho' in mod_dic.keys():
                    avg_cum = np.zeros((len(results[env_dic['name']][mod_dic['name']][0]['rho']),))
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
    if kwargs['anneal']:
        for env_dic in config['environment_params']:
            avg_cum = None
            for mod_dic in config['model_params']:
                if len(results[env_dic['name']][mod_dic['name']][0]['anneal']) > 0:
                    avg_cum = np.zeros((len(results[env_dic['name']][mod_dic['name']][0]['anneal']),))
                    for n in range(n_reps):
                        avg_cum += results[env_dic['name']][mod_dic['name']][n]['anneal']
                    avg_cum = avg_cum / n_reps
                    plt.xlabel('epochs')
                    plt.ylabel('anneal')
                    plt.plot(np.arange(len(avg_cum)), avg_cum, label=f"{mod_dic['name']} in {env_dic['name']}")
            if avg_cum is not None:
                plt.legend()
                plt.title('Annealing Coefficient over Training')
                plt.show()
    if kwargs['weights']:
        for env_dic in config['environment_params']:
            avg_cum = None
            for mod_dic in config['model_params']:
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
                        this_ax.plot(w_array[index], label=f'{index}', color=((jdx + 1) / len(w_indices), 0, 0, 0.5))
                        this_ax.set_xlabel('Epochs')
                        this_ax.set_ylabel('Weight Value')
                        this_ax.legend()
                        this_ax.set_title(f'{k} Weights')
                fig.suptitle(f"{mod_dic['name']}")
                plt.show()
    if kwargs['probabilities']:
        for env_dic in config['environment_params']:
            avg_cum = None
            for mod_dic in config['model_params']:
                p_values = results[env_dic['name']][mod_dic['name']][0]['probabilities']
                p_array = np.zeros_like(p_values)
                for n in range(n_reps):
                    p_array += results[env_dic['name']][mod_dic['name']][n]['probabilities']
                p_array = p_array / n_reps
                p_indices = list(product(*[range(d) for d in p_array.shape[:-1]]))
                for jdx, index in enumerate(p_indices):
                    plt.plot(p_array[index], label=f'{index}', color=((jdx + 1) / len(p_indices), 0, 0, 0.5))
                plt.xlabel('Epochs')
                plt.ylabel('Probability')
                plt.legend()
                plt.title(f"{mod_dic['name']} Probabilities")
                plt.show()
    if kwargs['success_probability']:
        for env_dic in config['environment_params']:
            for mod_dic in config['model_params']:
                p_values = results[env_dic['name']][mod_dic['name']][0]['probabilities']
                p_array = np.zeros_like(p_values)
                for n in range(n_reps):
                    p_array += results[env_dic['name']][mod_dic['name']][n]['probabilities']
                p_array = p_array / n_reps
                if env_dic['model'] == 'BanditTask':
                    p_success = p_array[0][env_dic['success_actions']]
                    plt.plot(p_success.transpose((1,0)), label=f'{mod_dic["name"]}')
            plt.xlabel('Epochs')
            plt.ylabel('Probability')
            plt.legend()
            plt.title(f"Success Probability in {env_dic['name']} ")
            plt.show()



# def learning_rates(simulator, results, n_reps, average=True, **kwargs):
#     for env_dic in simulator.environments:
#         if env_dic['name'] == 'GridWorld':
#             for mod_dic in simulator.models:
#                 if average:



if __name__ == "__main__":
    ...
