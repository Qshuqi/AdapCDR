# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time

from utils import ndcg_func,  recall_func, precision_func
acc_func = lambda x,y: np.sum(x == y) / len(x)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import expected_calibration_error

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x, is_training=False):
        if is_training:
            pred, u_emb, i_emb = self.forward(x, is_training)
            return pred.detach().cpu().numpy(), u_emb.detach().cpu().numpy(), i_emb.detach().cpu().numpy()
        else:
            pred = self.forward(x, is_training)
            return pred.detach().cpu().numpy()

class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.linear_1(z_emb))
        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class MF_adpt_cdr(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, n_bins = 15, calib_lamb = 1, calib_epoch_freq=2,
        num_epoch=1000, lr1=0.05, lr2=0.05, lr3=0.05, gamma = 0.1, lamb_prop = 0, lamb_pred = 0, lamb_imp = 0,
        tol=1e-4, G=1, option = 'ce', verbose=True): 

        optimizer_prop = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr1, weight_decay=lamb_prop)
        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr2, weight_decay=lamb_pred)
        optimizer_imp = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr3, weight_decay=lamb_imp)

        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                prop = self.propensity_model.forward(sub_x)
                inv_prop = 1/torch.clip(prop, gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x).cuda()                
                
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.forward(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction="sum")
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction="sum")
                        
                ips_loss = (xent_loss - imputation_loss)
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction="sum")
                
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                optimizer_pred.zero_grad()
                dr_loss.backward()
                optimizer_pred.step()

                pred_detached = pred.detach()
                e_loss = F.binary_cross_entropy(pred_detached, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred_detached, reduction="none")
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop.detach()).sum()
                optimizer_imp.zero_grad()
                imp_loss.backward()
                optimizer_imp.step()
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                prop_all = self.propensity_model.forward(x_sampled)

                if option == 'mse':
                    prop_loss = nn.MSELoss()(prop_all, sub_obs)
                else:
                    prop_loss = F.binary_cross_entropy(prop_all, sub_obs)
                # prop_loss = nn.MSELoss()(prop_all, sub_obs)
                
                loss = prop_loss

                if idx % int(calib_epoch_freq * total_batch) == 0:
                    calib_loss, boundaries = expected_calibration_error(sub_obs, prop_all, n_bins, return_boundaries=True)
                    loss += calib_lamb * calib_loss
                elif boundaries is not None:
                    calib_loss = expected_calibration_error(sub_obs, prop_all, n_bins, boundaries=boundaries)
                    loss += calib_lamb * calib_loss

                optimizer_prop.zero_grad()
                loss.backward()
                optimizer_prop.step()

                epoch_loss += dr_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL-ECE] epoch:{}, loss:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL-ECE] epoch:{}, loss:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL-ECE] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred