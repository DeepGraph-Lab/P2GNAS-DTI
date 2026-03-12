import argparse
import logging
import subprocess as sp

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os 

from function4testdata import sh_evaluation 
from utils import plot_heatmap

#################################
#################################

hidden_channels = 60
print(f'Embedding dimension {hidden_channels}')
FOLDER = 'Results/TEST_ACROSS_DATA_SETS' # output folder

datasets_o = ['DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
datasets_t = datasets_o

nreps_training = 3
nreps = 3

all_auc = {}  # {(rep_t, rep, train, eval): auc}
all_aupr = {} # {(rep_t, rep, train, eval): aupr}

for rep_t in tqdm(range(nreps_training), desc='Repetitions'):
    for database1 in datasets_o:
        print('======================')
        print(f'Working with {database1}')
        for database2 in datasets_t:
            if database2 != database1:
                database = f'{database1.upper()}_WO_{database2.upper()}'
            else:
                database = f'{database1}'
            print(database)
            try:
                print(f'database {database}, hidden_channels {hidden_channels}')
                return_code = sp.check_call(['python3', 'Code/main.py', '-d', f'{database}', '-e', f'{hidden_channels}'])
                if return_code == 0:
                    print(f'EXIT CODE 0 FOR {database.upper()} with hd {hidden_channels}')
            except sp.CalledProcessError as e:
                logging.info(e.output)
        print(f'Finished with {database1}')


    for rep in range(nreps):
        for dataset_trained in datasets_o:
            for dataset_evaluated in datasets_t:
                auc, aupr = sh_evaluation(DATABASE_TRAINED=dataset_trained, DATABASE_EVAL=dataset_evaluated, hd=hidden_channels)
                all_auc[(rep_t, rep, dataset_trained, dataset_evaluated)] = auc
                all_aupr[(rep_t, rep, dataset_trained, dataset_evaluated)] = aupr

print('finished!!!')

df_auc_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
df_aupr_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
df_auc_std_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)
df_aupr_std_hmp = pd.DataFrame(columns=datasets_o, index=datasets_t).astype(float)

for dataset_trained in datasets_o:
    for dataset_evaluated in datasets_t:
        auc_list, aupr_list = [], []
        for rep_t in range(nreps_training):
            for rep in range(nreps):
                auc = all_auc.get((rep_t, rep, dataset_trained, dataset_evaluated), np.nan)
                aupr = all_aupr.get((rep_t, rep, dataset_trained, dataset_evaluated), np.nan)
                auc_list.append(auc)
                aupr_list.append(aupr)
        final_auc = np.nanmean(auc_list)
        final_auc_std = np.nanstd(auc_list)
        final_aupr = np.nanmean(aupr_list)
        final_aupr_std = np.nanstd(aupr_list)
        df_auc_hmp.loc[dataset_evaluated, dataset_trained] = final_auc
        df_aupr_hmp.loc[dataset_evaluated, dataset_trained] = final_aupr
        df_auc_std_hmp.loc[dataset_evaluated, dataset_trained] = final_auc_std
        df_aupr_std_hmp.loc[dataset_evaluated, dataset_trained] = final_aupr_std

print(f"==== {hidden_channels} ====")
print('AUC')
print(df_auc_hmp)
print('AUPR')
print(df_aupr_hmp)

if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)

df_auc_hmp.to_csv(f'{FOLDER}/auc_{hidden_channels}.csv')
df_aupr_hmp.to_csv(f'{FOLDER}/aupr_{hidden_channels}.csv')
df_auc_std_hmp.to_csv(f'{FOLDER}/auc_std_{hidden_channels}.csv')
df_aupr_std_hmp.to_csv(f'{FOLDER}/aupr_std_{hidden_channels}.csv')

plot_heatmap(df_auc_hmp, df_auc_std_hmp, cmap='Oranges')
plt.savefig(f'{FOLDER}/results_AUC_{hidden_channels}.png', bbox_inches='tight', pad_inches=0.2)
plot_heatmap(df_aupr_hmp, df_aupr_std_hmp, 'Oranges')
plt.savefig(f'{FOLDER}/results_AUPR_{hidden_channels}.png', bbox_inches='tight', pad_inches=0.2)