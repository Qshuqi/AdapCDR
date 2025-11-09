# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb
import optuna
import time
import os
from dataset import load_data
from matrix_factorization_adpt_cdr import MF_adpt_cdr
from itertools import product
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU,recall_func, precision_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

dataset_name = 'kuai'

if dataset_name == "kuai":
    rdf_train = np.array(pd.read_table("./data/kuai/user.txt", header = None, sep = ','))     
    rdf_test = np.array(pd.read_table("./data/kuai/random.txt", header = None, sep = ','))
    rdf_train_new = np.c_[rdf_train, np.ones(rdf_train.shape[0])]
    rdf_test_new = np.c_[rdf_test, np.zeros(rdf_test.shape[0])]
    rdf = np.r_[rdf_train_new, rdf_test_new]
    
    rdf = rdf[np.argsort(rdf[:, 0])]
    c = rdf.copy()
    for i in range(rdf.shape[0]):
        if i == 0:
            c[:, 0][i] = i
            temp = rdf[:, 0][0]
        else:
            if c[:, 0][i] == temp:
                c[:, 0][i] = c[:, 0][i-1]
            else:
                c[:, 0][i] = c[:, 0][i-1] + 1
            temp = rdf[:, 0][i]
    
    c = c[np.argsort(c[:, 1])]
    d = c.copy()
    for i in range(rdf.shape[0]):
        if i == 0:
            d[:, 1][i] = i
            temp = c[:, 1][0]
        else:
            if d[:, 1][i] == temp:
                d[:, 1][i] = d[:, 1][i-1]
            else:
                d[:, 1][i] = d[:, 1][i-1] + 1
            temp = c[:, 1][i]

    y_train = d[:, 2][d[:, 3] == 1]
    y_test = d[:, 2][d[:, 3] == 0]
    x_train = d[:, :2][d[:, 3] == 1]
    x_test = d[:, :2][d[:, 3] == 0]
    
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1

    y_train = binarize(y_train, 2)
    y_test = binarize(y_test, 2)
    num_user = int(num_user)
    num_item = int(num_item)
        
def objective(trial):
        lamb1 = trial.suggest_categorical('lamb1', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        lamb2 = trial.suggest_categorical('lamb2', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        lamb3 = trial.suggest_categorical('lamb3', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        G = trial.suggest_int('G', 2, 5)  
        emb = trial.suggest_categorical('emb', [4,8,16]) 
        calib_lamb = trial.suggest_float('calib_lamb', 0.1, 10)
        gamma = trial.suggest_float('gamma', 0.01, 0.05)
        n_bins = trial.suggest_categorical('n_bins', [5,10,15,20,25])
        lr1 = trial.suggest_categorical('lr1', [0.01, 0.05])
        lr2 = trial.suggest_categorical('lr2', [0.01, 0.05])
        lr3 = trial.suggest_categorical('lr3', [0.01, 0.05])
        option = trial.suggest_categorical('option', ['ce', 'mse'])
        calib_epoch_freq = trial.suggest_categorical('calib_epoch_freq', [0.1, 0.2, 0.4, 0.6, 0.8, 1])

        model = MF_adpt_cdr(num_user, num_item, batch_size=2048, embedding_k = emb)
        model.cuda()
        model.fit(x_train, y_train, n_bins = n_bins, calib_lamb = calib_lamb, lr1 = lr1, lr2 = lr2, lr3 = lr3, calib_epoch_freq=calib_epoch_freq,
        gamma = gamma, lamb_prop=lamb1,
        lamb_pred=lamb2,
        lamb_imp=lamb3,
        G = G,
        option = option,
        tol=1e-5)
        
        test_pred = model.predict(x_test)
        mse_mfdrmc = mse_func(y_test, test_pred)
        auc_mfdrmc = roc_auc_score(y_test, test_pred)
        ndcg_res = ndcg_func(model, x_test, y_test, [20, 50])
        recall_res = recall_func(model, x_test, y_test, [20, 50])
        precision_res = precision_func(model, x_test, y_test, [20, 50])

        f1_20 = 2 * np.mean(recall_res["recall_20"]) * np.mean(precision_res["precision_20"]) / (np.mean(recall_res["recall_20"]) + np.mean(precision_res["precision_20"]))
        f1_50 = 2 * np.mean(recall_res["recall_50"]) * np.mean(precision_res["precision_50"]) / (np.mean(recall_res["recall_50"]) + np.mean(precision_res["precision_50"]))

        results = []
        results.append({'m': mse_mfdrmc})
        results.append({'a': auc_mfdrmc})
        results.append({'n20': np.mean(ndcg_res["ndcg_20"])})
        results.append({'r20': np.mean(recall_res["recall_20"])})
        results.append({'p20': np.mean(precision_res["precision_20"])})
        results.append({'n50': np.mean(ndcg_res["ndcg_50"])})
        results.append({'r50': np.mean(recall_res["recall_50"])})
        results.append({'p50': np.mean(precision_res["precision_50"])})
        results.append({'f1_20': f1_20})
        results.append({'f1_50': f1_50})

        with open('kuai2_icdmw26_ours_14n20_new.txt', 'a') as file:
                file.write(f'lamb1: {lamb1}, lamb2: {lamb2}, lamb3: {lamb3}, G: {G}, emb: {emb}, n_bins: {n_bins}, lr1: {lr1}, lr2: {lr2}, lr3: {lr3}, cali: {calib_lamb}, gamma: {gamma}, calib_epoch_freq: {calib_epoch_freq}, option: {option}' + '\n')
                for result in results:
                        for k, v in result.items():
                                file.write(k + ': ' + str(v.round(3)) + '\n')
                file.write('-----------------------------------' + '\n')
                file.write(str(auc_mfdrmc.round(3) + 1.4*np.mean(ndcg_res["ndcg_20"]).round(3)))
                file.write('\n')
                file.write('\n')
        return auc_mfdrmc + 1.4*np.mean(ndcg_res["ndcg_20"])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')  