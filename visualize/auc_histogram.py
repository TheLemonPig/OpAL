import pandas as pd
import matplotlib.pyplot as plt
import os

# prefix = ''
prefix = 'OpAL_'

datapath = f'{os.getcwd()}/Data/{prefix}AUCs_Compiled.csv'

try:
    compiled_aucs = pd.read_csv(datapath)[1:].to_numpy()
except FileNotFoundError:
    raise FileNotFoundError(f'No file called {datapath}')

if type(compiled_aucs[0,0]) == str:
    compiled_aucs = compiled_aucs[:,1:]

if prefix == 'OpAL_':
    avg_aucs = (compiled_aucs[:,:2] + compiled_aucs[:,2:]) / 2
    delta_auc = avg_aucs[:,1:] - avg_aucs[:,:1]
    print(delta_auc)
    plt.hist(delta_auc)
    plt.show()

