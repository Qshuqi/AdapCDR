"""Dataset loading utilities."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import os


def load_coat_dataset(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray]:
    """Load Coat dataset.
    
    Args:
        data_dir: Root directory containing data folder.
        
    Returns:
        Tuple of (train_matrix, test_matrix).
    """
    data_set_dir = os.path.join(data_dir, "coat")
    train_file = os.path.join(data_set_dir, "train.ascii")
    test_file = os.path.join(data_set_dir, "test.ascii")

    with open(train_file, "r") as f:
        x_train = []
        for line in f.readlines():
            x_train.append(line.split())
        x_train = np.array(x_train).astype(int)

    with open(test_file, "r") as f:
        x_test = []
        for line in f.readlines():
            x_test.append(line.split())
        x_test = np.array(x_test).astype(int)

    print(f"===>Load from coat data set<===")
    print(f"[train] rating ratio: {(x_train > 0).sum() / (x_train.shape[0] * x_train.shape[1]):.6f}")
    print(f"[test]  rating ratio: {(x_test > 0).sum() / (x_test.shape[0] * x_test.shape[1]):.6f}")

    return x_train, x_test


def load_yahoo_dataset(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Yahoo! R3 dataset.
    
    Args:
        data_dir: Root directory containing data folder.
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test).
    """
    data_set_dir = os.path.join(data_dir, "yahoo")
    train_file = os.path.join(data_set_dir, "ydata-ymusic-rating-study-v1_0-train.txt")
    test_file = os.path.join(data_set_dir, "ydata-ymusic-rating-study-v1_0-test.txt")

    x_train = []
    # Format: <user_id> <song id> <rating>
    with open(train_file, "r") as f:
        for line in f:
            x_train.append(line.strip().split())
    x_train = np.array(x_train).astype(int)

    x_test = []
    # Format: <user_id> <song id> <rating>
    with open(test_file, "r") as f:
        for line in f:
            x_test.append(line.strip().split())
    x_test = np.array(x_test).astype(int)
    
    print(f"===>Load from yahoo data set<===")
    print(f"[train] num data: {x_train.shape[0]}")
    print(f"[test]  num data: {x_test.shape[0]}")

    return x_train[:, :-1], x_train[:, -1], x_test[:, :-1], x_test[:, -1]


def load_kuai_dataset(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load KuaiRec dataset.
    
    Args:
        data_dir: Root directory containing data folder.
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test).
    """
    train_file = os.path.join(data_dir, "kuai", "user.txt")
    test_file = os.path.join(data_dir, "kuai", "random.txt")
    
    rdf_train = np.array(pd.read_table(train_file, header=None, sep=','))
    rdf_test = np.array(pd.read_table(test_file, header=None, sep=','))
    
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
                c[:, 0][i] = c[:, 0][i - 1]
            else:
                c[:, 0][i] = c[:, 0][i - 1] + 1
            temp = rdf[:, 0][i]

    c = c[np.argsort(c[:, 1])]
    d = c.copy()
    for i in range(rdf.shape[0]):
        if i == 0:
            d[:, 1][i] = i
            temp = c[:, 1][0]
        else:
            if d[:, 1][i] == temp:
                d[:, 1][i] = d[:, 1][i - 1]
            else:
                d[:, 1][i] = d[:, 1][i - 1] + 1
            temp = c[:, 1][i]

    y_train = d[:, 2][d[:, 3] == 1]
    y_test = d[:, 2][d[:, 3] == 0]
    x_train = d[:, :2][d[:, 3] == 1]
    x_test = d[:, :2][d[:, 3] == 0]

    print(f"===>Load from kuai data set<===")
    print(f"[train] num data: {x_train.shape[0]}")
    print(f"[test]  num data: {x_test.shape[0]}")

    return x_train, y_train, x_test, y_test


def load_data(name: str = "coat", data_dir: str = "./data"):
    """Load dataset by name.
    
    Args:
        name: Dataset name ('coat', 'yahoo', or 'kuai').
        data_dir: Root directory containing data folder.
        
    Returns:
        For 'coat': (train_matrix, test_matrix)
        For 'yahoo' or 'kuai': (x_train, y_train, x_test, y_test)
    """
    if name == "coat":
        return load_coat_dataset(data_dir)
    elif name == "yahoo":
        return load_yahoo_dataset(data_dir)
    elif name == "kuai":
        return load_kuai_dataset(data_dir)
    else:
        raise ValueError(f"Unknown dataset name: {name}. Supported: 'coat', 'yahoo', 'kuai'")

