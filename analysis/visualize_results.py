import matplotlib.pyplot as plt
import numpy as np

from utils import location_counter


def state_heatmap(results, n_reps, **kwargs):
    for simulator_dic in results:
        domain = simulator_dic['simulator'].model.domain
        if kwargs['average']:
            location_counts = np.zeros(domain)
            for n in range(n_reps):
                state_list = simulator_dic['states'][n]
                location_counts += location_counter(state_list, domain) / n_reps
            plt.imshow(location_counts, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.show()
        else:
            x, y = domain
            location_counts = np.zeros((x, y, n_reps))
            for n in range(n_reps):
                state_list = simulator_dic['states'][n]
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




if __name__ == "__main__":
    example()
