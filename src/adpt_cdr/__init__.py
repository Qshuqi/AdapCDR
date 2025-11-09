"""Adaptive Calibrated Doubly Robust Estimators for Debiased Recommendation."""

__version__ = "1.0.0"

from adpt_cdr.models import MF_adpt_cdr, MF_BaseModel, NCF_BaseModel
from adpt_cdr.datasets import load_data, load_coat_dataset, load_yahoo_dataset, load_kuai_dataset
from adpt_cdr.utils import (
    rating_mat_to_sample,
    binarize,
    shuffle,
    ndcg_func,
    recall_func,
    precision_func,
    expected_calibration_error,
    get_user_wise_ctr,
    gini_index,
)

__all__ = [
    "MF_adpt_cdr",
    "MF_BaseModel",
    "NCF_BaseModel",
    "load_data",
    "load_coat_dataset",
    "load_yahoo_dataset",
    "load_kuai_dataset",
    "rating_mat_to_sample",
    "binarize",
    "shuffle",
    "ndcg_func",
    "recall_func",
    "precision_func",
    "expected_calibration_error",
    "get_user_wise_ctr",
    "gini_index",
]
