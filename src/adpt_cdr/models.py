"""Matrix Factorization models for Adaptive Calibrated Doubly Robust estimation."""

from typing import Tuple, Optional
import numpy as np
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as F

from adpt_cdr.utils import expected_calibration_error


def generate_total_sample(num_user: int, num_item: int) -> np.ndarray:
    """Generate all user-item pairs.
    
    Args:
        num_user: Number of users.
        num_item: Number of items.
        
    Returns:
        Array of shape (num_user * num_item, 2) containing all user-item pairs.
    """
    sample = []
    for i in range(num_user):
        sample.extend([[i, j] for j in range(num_item)])
    return np.array(sample)


class MF_BaseModel(nn.Module):
    """Matrix Factorization base model.
    
    This model learns user and item embeddings and predicts ratings
    using the dot product of embeddings.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_k: int = 4, **kwargs):
        """Initialize MF base model.
        
        Args:
            num_users: Number of users.
            num_items: Number of items.
            embedding_k: Embedding dimension.
        """
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = nn.Embedding(self.num_users, self.embedding_k)
        self.H = nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = nn.Sigmoid()
        self.xent_func = nn.BCELoss()
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: np.ndarray, is_training: bool = False) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input array of shape (batch_size, 2) containing [user_idx, item_idx].
            is_training: Whether in training mode.
            
        Returns:
            If is_training: (predictions, user_embeddings, item_embeddings)
            Otherwise: predictions only
        """
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Predict ratings.
        
        Args:
            x: Input array of shape (batch_size, 2).
            is_training: Whether in training mode.
            
        Returns:
            Predictions as numpy array.
        """
        if is_training:
            pred, u_emb, i_emb = self.forward(x, is_training)
            return pred.detach().cpu().numpy(), u_emb.detach().cpu().numpy(), i_emb.detach().cpu().numpy()
        else:
            pred = self.forward(x, is_training)
            return pred.detach().cpu().numpy()


class NCF_BaseModel(nn.Module):
    """Neural Collaborative Filtering base model.
    
    Uses a neural network to combine user and item embeddings.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_k: int = 4, **kwargs):
        """Initialize NCF base model.
        
        Args:
            num_users: Number of users.
            num_items: Number of items.
            embedding_k: Embedding dimension.
        """
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = nn.Embedding(self.num_users, self.embedding_k)
        self.H = nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = nn.Linear(self.embedding_k * 2, 1, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.xent_func = nn.BCELoss()
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: np.ndarray, is_training: bool = False) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input array of shape (batch_size, 2).
            is_training: Whether in training mode.
            
        Returns:
            If is_training: (predictions, user_embeddings, item_embeddings)
            Otherwise: predictions only
        """
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # Concatenate embeddings
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.linear_1(z_emb))
        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)

    def predict(self, x: np.ndarray) -> torch.Tensor:
        """Predict ratings.
        
        Args:
            x: Input array of shape (batch_size, 2).
            
        Returns:
            Predictions as torch tensor.
        """
        pred = self.forward(x)
        return pred.detach().cpu()


class MF_adpt_cdr(nn.Module):
    """Adaptive Calibrated Doubly Robust Matrix Factorization model.
    
    This model combines prediction, imputation, and propensity models
    to debias recommendations using doubly robust estimation with
    adaptive calibration.
    """
    
    def __init__(self, num_users: int, num_items: int, batch_size: int, embedding_k: int = 4, **kwargs):
        """Initialize MF_adpt_cdr model.
        
        Args:
            num_users: Number of users.
            num_items: Number of items.
            batch_size: Batch size for training.
            embedding_k: Embedding dimension.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_k=self.embedding_k,
            **kwargs
        )
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_k=self.embedding_k
        )
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_k=self.embedding_k,
            **kwargs
        )

        self.sigmoid = nn.Sigmoid()
        self.xent_func = nn.BCELoss()
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure all sub-models use the same device
        self.prediction_model.device = self.device
        self.imputation_model.device = self.device
        self.propensity_model.device = self.device

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        stop: int = 5,
        n_bins: int = 15,
        calib_lamb: float = 1.0,
        calib_epoch_freq: float = 2.0,
        num_epoch: int = 1000,
        lr1: float = 0.05,
        lr2: float = 0.05,
        lr3: float = 0.05,
        gamma: float = 0.1,
        lamb_prop: float = 0.0,
        lamb_pred: float = 0.0,
        lamb_imp: float = 0.0,
        tol: float = 1e-4,
        G: int = 1,
        option: str = 'ce',
        verbose: bool = True
    ) -> None:
        """Train the model.
        
        Args:
            x: Training features of shape (n_samples, 2) with [user_idx, item_idx].
            y: Training labels of shape (n_samples,).
            stop: Early stopping patience.
            n_bins: Number of bins for calibration error calculation.
            calib_lamb: Weight for calibration loss.
            calib_epoch_freq: Frequency of calibration error calculation.
            num_epoch: Maximum number of training epochs.
            lr1: Learning rate for propensity model.
            lr2: Learning rate for prediction model.
            lr3: Learning rate for imputation model.
            gamma: Clipping threshold for inverse propensity.
            lamb_prop: L2 regularization for propensity model.
            lamb_pred: L2 regularization for prediction model.
            lamb_imp: L2 regularization for imputation model.
            tol: Tolerance for early stopping.
            G: Multiplier for sampling unlabeled data.
            option: Loss function option ('ce' or 'mse') for propensity model.
            verbose: Whether to print training progress.
        """
        optimizer_prop = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr1, weight_decay=lamb_prop)
        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr2, weight_decay=lamb_pred)
        optimizer_imp = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr3, weight_decay=lamb_imp)

        last_loss = 1e9
        obs = sps.csr_matrix(
            (np.ones(len(y)), (x[:, 0], x[:, 1])),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        ).toarray().reshape(-1)

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        boundaries = None

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size * idx:(idx + 1) * self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                prop = self.propensity_model.forward(sub_x)
                inv_prop = 1 / torch.clip(prop, gamma, 1)

                sub_y = torch.Tensor(sub_y).to(self.device)

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x).to(self.device)

                x_all_idx = ul_idxs[G * idx * self.batch_size:G * (idx + 1) * self.batch_size]
                x_sampled = x_all[x_all_idx]

                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.forward(x_sampled).to(self.device)

                xent_loss = F.binary_cross_entropy(
                    pred, sub_y, weight=inv_prop.detach(), reduction="sum")
                imputation_loss = F.binary_cross_entropy(
                    pred, imputation_y.detach(), reduction="sum")

                ips_loss = (xent_loss - imputation_loss)

                direct_loss = F.binary_cross_entropy(
                    pred_u, imputation_y1.detach(), reduction="sum")

                dr_loss = (ips_loss + direct_loss) / x_sampled.shape[0]
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

                sub_obs = torch.Tensor(obs[x_all_idx]).to(self.device)
                prop_all = self.propensity_model.forward(x_sampled)

                if option == 'mse':
                    prop_loss = nn.MSELoss()(prop_all, sub_obs)
                else:
                    prop_loss = F.binary_cross_entropy(prop_all, sub_obs)

                loss = prop_loss

                if idx % int(calib_epoch_freq * total_batch) == 0:
                    calib_loss, boundaries = expected_calibration_error(
                        sub_obs, prop_all, n_bins, return_boundaries=True)
                    loss += calib_lamb * calib_loss
                elif boundaries is not None:
                    calib_loss = expected_calibration_error(
                        sub_obs, prop_all, n_bins, boundaries=boundaries)
                    loss += calib_lamb * calib_loss

                optimizer_prop.zero_grad()
                loss.backward()
                optimizer_prop.step()

                epoch_loss += dr_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print(f"[MF-DR-JL-ECE] epoch:{epoch}, loss:{epoch_loss}")
                    break
                else:
                    early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print(f"[MF-DR-JL-ECE] epoch:{epoch}, loss:{epoch_loss}")

            if epoch == num_epoch - 1:
                if verbose:
                    print("[MF-DR-JL-ECE] Reach preset epochs, it seems does not converge.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict ratings for given user-item pairs.
        
        Args:
            x: Input array of shape (n_samples, 2) with [user_idx, item_idx].
            
        Returns:
            Predictions as numpy array of shape (n_samples,).
        """
        pred = self.prediction_model.predict(x)
        return pred

