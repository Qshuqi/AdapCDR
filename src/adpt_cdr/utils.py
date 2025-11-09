"""Utility functions for evaluation and data processing."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import auc


def rating_mat_to_sample(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert rating matrix to sample format.
    
    Args:
        mat: Rating matrix of shape (num_users, num_items).
        
    Returns:
        Tuple of (x, y) where:
            x: Array of shape (n_samples, 2) with [user_idx, item_idx]
            y: Array of shape (n_samples,) with ratings
    """
    row, col = np.nonzero(mat)
    y = mat[row, col]
    x = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)
    return x, y


def binarize(y: np.ndarray, thres: float = 3.0) -> np.ndarray:
    """Binarize ratings using a threshold.
    
    Args:
        y: Ratings array.
        thres: Threshold value. Ratings >= thres become 1, others become 0.
        
    Returns:
        Binarized array.
    """
    y = y.copy()
    y[y < thres] = 0
    y[y >= thres] = 1
    return y


def shuffle(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle data arrays together.
    
    Args:
        x: Feature array.
        y: Label array.
        
    Returns:
        Tuple of shuffled (x, y).
    """
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]


def ndcg_func(
    model,
    x_te: np.ndarray,
    y_te: np.ndarray,
    top_k_list: List[int] = [5, 10]
) -> Dict[str, List[float]]:
    """Evaluate nDCG@K of the trained model on test dataset.
    
    Args:
        model: Model with predict method.
        x_te: Test features of shape (n_samples, 2).
        y_te: Test labels of shape (n_samples,).
        top_k_list: List of K values for nDCG@K.
        
    Returns:
        Dictionary mapping "ndcg_{k}" to list of nDCG@k values per user.
    """
    all_user_idx = np.unique(x_te[:, 0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:, 0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]

            log2_iplus1 = np.log2(1 + np.arange(1, top_k + 1))

            dcg_k = y_u[pred_top_k] / log2_iplus1
            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1.0
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map[f"ndcg_{top_k}"].append(ndcg_k)

    return result_map


def recall_func(
    model,
    x_te: np.ndarray,
    y_te: np.ndarray,
    top_k_list: List[int] = [5, 10]
) -> Dict[str, List[float]]:
    """Evaluate Recall@K of the trained model on test dataset.
    
    Args:
        model: Model with predict method.
        x_te: Test features of shape (n_samples, 2).
        y_te: Test labels of shape (n_samples,).
        top_k_list: List of K values for Recall@K.
        
    Returns:
        Dictionary mapping "recall_{k}" to list of Recall@k values per user.
    """
    all_user_idx = np.unique(x_te[:, 0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:, 0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            recall = np.sum(y_u[pred_top_k]) / max(1, min(np.sum(y_u), top_k))
            result_map[f"recall_{top_k}"].append(recall)

    return result_map


def precision_func(
    model,
    x_te: np.ndarray,
    y_te: np.ndarray,
    top_k_list: List[int] = [5, 10]
) -> Dict[str, List[float]]:
    """Evaluate Precision@K of the trained model on test dataset.
    
    Args:
        model: Model with predict method.
        x_te: Test features of shape (n_samples, 2).
        y_te: Test labels of shape (n_samples,).
        top_k_list: List of K values for Precision@K.
        
    Returns:
        Dictionary mapping "precision_{k}" to list of Precision@k values per user.
    """
    all_user_idx = np.unique(x_te[:, 0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:, 0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()
            if min(count, top_k) != 0:
                precision = np.sum(y_u[pred_top_k]) / top_k
            else:
                precision = 0.0
            result_map[f"precision_{top_k}"].append(precision)

    return result_map


def expected_calibration_error(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    n_bins: int = 15,
    boundaries: Optional[torch.Tensor] = None,
    return_boundaries: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration.
        boundaries: Pre-computed bin boundaries (optional).
        return_boundaries: Whether to return computed boundaries.
        
    Returns:
        ECE value, and optionally boundaries if return_boundaries=True.
    """
    y_prob = y_prob.to(torch.float32)
    y_true = y_true.to(torch.float32)

    n_samples = len(y_prob)

    sorted_probs, sorted_indices = torch.sort(y_prob)
    sorted_labels = y_true[sorted_indices]

    if boundaries is None:
        # Generate candidate split points from quantiles
        quantiles = torch.linspace(0.01, 0.99, 50, device=y_prob.device)
        candidate_boundaries = torch.quantile(y_prob, quantiles)
        candidate_boundaries = torch.unique(candidate_boundaries)

        # Convert probability boundaries to indices in the sorted array
        candidate_split_indices = torch.searchsorted(sorted_probs, candidate_boundaries).tolist()
        candidate_split_indices = sorted(list(set(candidate_split_indices)))

        # Helper function to calculate ECE for a slice of data
        def _calculate_ece_for_slice(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            if len(probs) == 0:
                return torch.tensor(0.0, device=probs.device)
            accuracy = torch.mean(labels.float())
            confidence = torch.mean(probs.float())
            return torch.abs(accuracy - confidence) * len(probs)

        # Greedy search for the best bins using candidate splits
        splits = [0, n_samples]

        for _ in range(n_bins - 1):
            current_max_increase = -1
            best_split_point = -1
            best_bin_index = -1

            for i in range(len(splits) - 1):
                start_idx, end_idx = splits[i], splits[i + 1]
                if end_idx - start_idx <= 1:
                    continue

                original_ece_contrib = _calculate_ece_for_slice(
                    sorted_probs[start_idx:end_idx],
                    sorted_labels[start_idx:end_idx]
                )

                # Find candidate splits within the current bin
                possible_splits = [
                    idx for idx in candidate_split_indices
                    if start_idx < idx < end_idx
                ]

                for split_idx in possible_splits:
                    ece1 = _calculate_ece_for_slice(
                        sorted_probs[start_idx:split_idx],
                        sorted_labels[start_idx:split_idx]
                    )
                    ece2 = _calculate_ece_for_slice(
                        sorted_probs[split_idx:end_idx],
                        sorted_labels[split_idx:end_idx]
                    )
                    increase = (ece1 + ece2) - original_ece_contrib

                    if increase > current_max_increase:
                        current_max_increase = increase
                        best_split_point = split_idx
                        best_bin_index = i

            if best_split_point != -1:
                splits.insert(best_bin_index + 1, best_split_point)
                splits.sort()
            else:
                break  # No more beneficial splits found

        # Convert splits to probability boundaries
        boundaries = []
        for split in splits[1:-1]:  # exclude 0 and n_samples
            if split > 0 and split < n_samples:
                boundaries.append(sorted_probs[split].item())
        boundaries = torch.tensor(boundaries, device=y_prob.device)
    else:
        # Use provided boundaries to determine splits
        splits = [0] + torch.searchsorted(sorted_probs, boundaries).tolist() + [n_samples]
        splits = sorted(list(set(splits)))  # Ensure unique and sorted

    # Calculate final ECE based on the determined splits
    total_ece = torch.tensor(0.0, device=y_prob.device, requires_grad=True)
    for i in range(len(splits) - 1):
        start_idx, end_idx = splits[i], splits[i + 1]
        bin_probs = sorted_probs[start_idx:end_idx]
        bin_labels = sorted_labels[start_idx:end_idx]
        if len(bin_probs) > 0:
            accuracy = torch.mean(bin_labels.float())
            confidence = torch.mean(bin_probs.float())
            total_ece = total_ece + torch.abs(accuracy - confidence) * len(bin_probs)

    ece = total_ece / n_samples

    if return_boundaries:
        return ece, boundaries
    else:
        return ece


def get_user_wise_ctr(
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
    top_N: int = 5
) -> np.ndarray:
    """Calculate user-wise CTR (Click-Through Rate).
    
    Args:
        x_test: Test features of shape (n_samples, 2).
        y_test: Test labels of shape (n_samples,).
        test_pred: Test predictions of shape (n_samples,).
        top_N: Number of top items to consider.
        
    Returns:
        Array of CTR values per user.
    """
    offset = 0
    user_idxs = np.unique(x_test[:, 0])
    user_ctr_list = []
    for user in user_idxs:
        mask = x_test[:, 0] == user
        pred_item = np.argsort(-test_pred[mask])[:top_N] + offset
        u_ctr = y_test[pred_item].sum() / pred_item.shape[0]
        user_ctr_list.append(u_ctr)
        offset += mask.sum()

    return np.array(user_ctr_list)


def gini_index(user_utility: np.ndarray) -> Tuple[float, float]:
    """Calculate Gini index and global utility.
    
    Args:
        user_utility: Array of utility values per user.
        
    Returns:
        Tuple of (gini_index, global_utility).
    """
    cum_L = np.cumsum(np.sort(user_utility))
    sum_L = np.sum(user_utility)
    num_user = len(user_utility)
    xx = np.linspace(0, 1, num_user + 1)
    yy = np.append([0], cum_L / sum_L)

    gi = (0.5 - auc(xx, yy)) / 0.5
    gu = sum_L / num_user

    print(f"Num User: {num_user}")
    print(f"Gini index: {gi}")
    print(f"Global utility: {gu}")
    return gi, gu

