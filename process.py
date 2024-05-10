import os
import pickle
import numpy as np
import pandas as pd
from itertools import product

from analysis.quantify_results import success_metrics

from_csv = True
opal_data = True

# preprefix = 'AUCs_'
preprefix = 'Test_'

if opal_data:
    prefix = 'OpAL_'
    #cols = 'OpALPlus/GridWorldSmallSparse,OpALPlus/GridWorldSmallRich,OpALPlus/GridWorldLargeSparse,OpALPlus/GridWorldLargeRich,OpALStar/GridWorldSmallSparse,OpALStar/GridWorldSmallRich,OpALStar/GridWorldLargeSparse,OpALStar/GridWorldLargeRich'.split(',')
else:
    prefix = ''
    #cols = 'ActorCritic/GridWorldSmallSparse,ActorCritic/GridWorldSmallRich,ActorCritic/GridWorldLargeSparse,ActorCritic/GridWorldLargeRich,QLearning/GridWorldSmallSparse,QLearning/GridWorldSmallRich,QLearning/GridWorldLargeSparse,QLearning/GridWorldLargeRich'.split(',')

if from_csv:
    if preprefix.startswith('AUC'):
        file_prefix = f'{prefix}{preprefix}'
    else:
        file_prefix = f'{prefix}{preprefix}Rates_'
else:
    file_prefix = f'{prefix}Results_'

log_folder = f'{prefix}Logs'
data_folder = f'Data'
out_data_folder = f'Compiled_Data'
log_path = os.path.join(os.getcwd(),log_folder)
data_path = os.path.join(os.getcwd(),data_folder)
out_data_path = os.path.join(os.getcwd(),out_data_folder)

path_prefix = os.path.join(log_folder,file_prefix)
config_path = os.path.join(log_folder,f'{prefix}Config.pkl')

log = {}
config = None
hyperparams = None
found = False
max_seed_value = 0

for file in os.listdir(log_path):
    if not file.endswith('Config.pkl'):
        seed_value = int(file[len(file_prefix):-4])
        max_seed_value = max(seed_value, max_seed_value)
for file in os.listdir(data_path):
    if not file.endswith('Config.pkl'):
        if file.startswith(file_prefix) and not file.endswith('Compiled.csv'):
            seed_value = int(file[len(file_prefix):-4])
            max_seed_value = max(seed_value, max_seed_value)
        # if file.startswith(file_prefix+'0'):
        #     assert not found
        #     if from_csv:
        #         filepath = os.path.join(data_path,file)
        #         df = pd.read_csv(filepath)
        #     else:
        #         filepath = os.path.join(log_path,file)
        #         with open(filepath, 'rb') as f:
        #             results = pickle.load(f)
        #     found = True


with open(config_path, 'rb') as f:
    config = pickle.load(f)
hyperparams = config['hyperparams']
lists_of_hyperparams = dict()
for k,v in hyperparams.items():
    if type(v) == tuple:
        start, stop, step = v
        stop += 1e-10
        sublist = np.arange(start, stop, step)
    elif type(v) == list:
        sublist = v
    lists_of_hyperparams.update({k: np.round(sublist,decimals=5)})
param_permutations = list(product(*lists_of_hyperparams.values()))
n_permutations = len(param_permutations)
n_models = len(config['model_params'])
n_envs = len(config['environment_params'])
np_results = np.zeros((n_permutations,n_envs*n_models))
n_contexts = n_models * n_envs
success_rates = np.zeros((n_models,n_envs,n_permutations)) #.transpose((1,0,2))
success_rates_list = []
df_index = []
cols = None
if from_csv:
    for file in os.listdir(data_folder):
        if file.startswith(f'{prefix}{preprefix}') and not file.endswith('Compiled.csv'):
            #print(f'Opening {file}') 

            filepath = os.path.join(data_folder,file)
            results = pd.read_csv(filepath)
            if cols is None:
                cols = results.columns
            else:
                assert (cols == results.columns).all(), "Column Mismatch -- Probably some slurm tasks did not finish in time"
            np_results += np.nan_to_num(results.to_numpy()) / (max_seed_value+1)
            success_rates += np.nan_to_num(results.to_numpy()).reshape((1,-1)).reshape((n_models,n_envs,n_permutations), order='f')/ (max_seed_value+1)# .transpose((1,0,2)) / (max_seed_value+1)
            #print(np.nan_to_num(results.to_numpy())[0])
            success_rates_list.append(np.nan_to_num(results.to_numpy()).reshape((1,-1)).reshape((n_models,n_envs,n_permutations), order='f')) #.transpose((1,0,2)))

df_results = pd.DataFrame(data=np_results,columns=cols,index=param_permutations)
# else:
#     for file in os.listdir(log_folder):
#         print(f'Opening {file}')  
#         if file.endswith('Config.pkl'):
#             filepath = os.path.join(log_folder,file)
#             with open(filepath,'rb') as f:
#                 results = pickle.load(f)
#             for idx, result_for_param in enumerate(results):
#                 print(result_for_param)
#                 success_rate = np.array(success_metrics(config, result_for_param, 1, verbose=False))  / (max_seed_value + 1)

#                 print(success_rate)
#                 for i in range(n_models):
#                     for j in range(n_envs):
#                         success_rates[i,j,idx] = success_rate[i * n_envs + j]

best_params_within_arg = np.argmax(success_rates, axis=2)
best_params_across_arg = np.argmax(success_rates.mean(axis=1), axis=1)
for i in range(n_envs):
    env_name = config['environment_params'][i]['name']
    print(f'\nBest Params within {env_name} for ...')
    for j in range(n_models):
        model_name = config['model_params'][j]['name']
        best_params_within = param_permutations[best_params_within_arg[j,i]]
        bpw_success = np.round(success_rates[j,i,best_params_within_arg[j,i]],decimals=2)
        bpw_std  = np.round(np.array([success_rates_list[n][j,i,best_params_within_arg[j,i]] for n in range(max_seed_value)]).std()/np.sqrt(max_seed_value),decimals=2)
        print(f'{model_name}: {tuple(np.round(param, decimals=5) for param in best_params_within)} {np.round(bpw_success,decimals=2)}% +/- {bpw_std}%')
        #print(f'arg: {best_params_within_arg[i,j]}')

for j in range(n_models):
    model_name = config['model_params'][j]['name']
    best_params_across = param_permutations[best_params_across_arg[j]]
    bpa_success = np.round(success_rates[j,:,best_params_across_arg[j]].mean(),decimals=2) #.max()
    bpa_std  = np.round(np.array([success_rates_list[n][j,:,best_params_across_arg[j]] for n in range(max_seed_value)]).max(axis=1).std()/np.sqrt(max_seed_value),decimals=2)
    print(f'\nBest Params Across environments for {model_name}: {tuple(np.round(param, decimals=5) for param in best_params_across)}  {bpa_success}% +/- {bpa_std}%')
    

success_array = np.round(success_rates.reshape((success_rates.shape[0]*success_rates.shape[1],-1)),decimals=2)
df = pd.DataFrame(data={results.columns[i]: success_array[i] for i in range(len(results.columns))},index=param_permutations)
if from_csv:
    path = os.path.join(out_data_path,f'{file_prefix}Compiled.csv')
    # np.savetxt(path,success_array,delimiter=',')
    df_results.to_csv(path)

# print(df_results.fillna(0).max(axis=0))

# df = pd.DataFrame(all_results)

# print(f'Saving Data')
# df.to_csv(data_path)

    



# std_results = std_results.std(axis=1) / np.sqrt(max_seed_value)
# all_results = {}
# for key in mean_results.keys():
#     all_results[key+'_mean'] = mean_results[key]
#     all_results[key+'_std'] = std_results[key]
# mean_results[key] = np.zeros(config['epochs'])
# std_results[key] = np.zeros(config['epochs'], max_seed_value)    
# for file in os.listdir(folder):
#     print(f'Opening {file}')
#     with open(os.path.join(folder,file),'rb') as f:
#         log = pickle.load(f)
#     log_results = log['results']
#     seed_value = int(file[path_prefix,path_prefix+1])
#         for key in log_results.keys():
#         result = log_results[key]
#         mean_results[key] += (np.array(result) / max_seed_value)
#         std_results[key,seed_value] += np.array(result)
