import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os

# prefix = ''
prefix = 'OpAL_'
current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
datapath = f'{parent_directory}/OpAL/Compiled_Data/{prefix}AUCs_Compiled.csv'

try:
    compiled_aucs = pd.read_csv(datapath)[1:].to_numpy()
except FileNotFoundError:
    raise FileNotFoundError(f'No file called {datapath}')

if type(compiled_aucs[0,0]) == str:
    compiled_aucs = compiled_aucs[:,1:]

if prefix == 'OpAL_':
    lean_aucs = compiled_aucs[:,:2]
    rich_aucs = compiled_aucs[:,2:]

    fig, ax = plt.subplots(2, 1)
    # 

    lean_delta_auc = lean_aucs[:, :1] - lean_aucs[:, 1:]
    ax[0].hist(lean_delta_auc, color='purple')
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel('AUC (Lean)')

    rich_delta_auc = rich_aucs[:, :1] - rich_aucs[:, 1:]
    ax[1].hist(rich_delta_auc, color='purple')
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel('AUC (Rich)')
    plt.suptitle('OpAL* - OpAL+ AUC')
    plt.show()

    fig, ax = plt.subplots(2, 1)

    ax[0].scatter(lean_aucs[:, 1:], lean_delta_auc, c='grey', s=0.5)
    ax[0].set_ylabel('OpAL* - OpAL+ AUC')
    ax[0].set_xlabel('OpAL+ AUC (Lean)')
    ax[0].axhline(0, c='lightgrey')

    ax[1].scatter(rich_aucs[:, 1:], rich_delta_auc, c='grey', s=0.5)
    ax[1].set_ylabel('OpAL* - OpAL+ AUC')
    ax[1].set_xlabel('OpAL+ AUC (Rich)')
    ax[1].axhline(0, c='lightgrey')
    plt.suptitle('OpAL* - OpAL+ vs OpAL+ AUC')
    plt.show()
    