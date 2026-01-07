"""
Dataset wrappers for DARPA flows (windowed inputs) for PyTorch and DeepProbLog.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from deepproblog.dataset import Dataset as DPLDataset
from deepproblog.query import Query
from problog.logic import Term, Constant

# Root directory for dataset files
ROOT_DIR = Path(__file__).parent

# Load datasets once

datasets_data = {
    "train": np.load(ROOT_DIR / "processed/X_train.npy", allow_pickle=True),
    "test":  np.load(ROOT_DIR / "processed/X_test.npy", allow_pickle=True),
}
# datasets_labels = {
#     "train": np.load(ROOT_DIR / "processed/y_train_binary.npy", allow_pickle=True),
#     "test":  np.load(ROOT_DIR / "processed/y_test_binary.npy", allow_pickle=True),
# }

phase = 1
# datasets_data = {
#     "train": np.load(ROOT_DIR / "processed/X_train.npy", allow_pickle=True),
#     "test":  np.load(ROOT_DIR / f"processed/X_phase_{phase}_attack_test.npy", allow_pickle=True),
# }
# datasets_labels = {
#     "train": np.load(ROOT_DIR / f"processed/y_phase_{phase}_train.npy", allow_pickle=True),
#     "test":  np.load(ROOT_DIR / f"processed/y_phase_{phase}_attack_test.npy", allow_pickle=True),
# }
datasets_labels = {
    "train": np.load(ROOT_DIR / f"processed/y_phase_{phase}_train.npy", allow_pickle=True),
    "test":  np.load(ROOT_DIR / f"processed/y_phase_{phase}_test.npy", allow_pickle=True),
}


class DARPAWindowedDataset(torch.utils.data.Dataset):
    """PyTorch dataset for LSTM pretraining (returns X, y)."""

    def __init__(self, X, y):
        dense_windows = []
        for _, w in enumerate(X):
            # Convert sparse to dense
            if sp.issparse(w):
                dense_windows.append(w.toarray())
            else:
                dense_windows.append(np.asarray(w, dtype=np.float32))

        # Stack into (N, seq_len, n_features)
        self.X = torch.from_numpy(np.stack(dense_windows).astype(np.float32))
        # Use long for cross-entropy loss
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FlowTensorSource(torch.utils.data.Dataset):
    """Tensor source for DeepProbLog (returns X only)."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        X = datasets_data[dataset_name]

        dense_windows = []
        for i, w in enumerate(X):
            if sp.issparse(w):
                dense_windows.append(w.toarray())
            else:
                dense_windows.append(np.asarray(w, dtype=np.float32))

        self.X = torch.from_numpy(np.stack(dense_windows).astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Handle DeepProbLog index conventions

        # Handle (Constant(i),)
        if isinstance(idx, tuple):
            if len(idx) == 0:
                raise RuntimeError(
                    "Empty index received. Usually means tensor accessed without example index."
                )
            idx = idx[0]

        # Handle Constant(i)
        if isinstance(idx, Constant):
            idx = int(idx.value)

        # print(f"[DARPAWindowed] Fetching index: {idx}")
        tensor = self.X[idx]
        # print(f"[DARPAWindowed] Tensor shape: {tensor.shape}")

        return tensor
    
    
class DARPADPLDataset(DPLDataset):
    def __init__(
            self, 
            dataset_name: str,
            function_name: str,
        ):
        """
        Dataset of Prolog queries for DeepProbLog.

        :param dataset_name: Dataset to use ("train" or "test")
        :param function_name: Name of Problog function to query
        """

        super().__init__()
        assert dataset_name in ["train", "test"]
        self.dataset_name = dataset_name
        self.y = datasets_labels[self.dataset_name]
        self.function_name = function_name

    def __len__(self):
        return len(self.y)

    def to_query(self, i):
        """Return a Query object for the i-th example."""
        label = int(self.y[i])
        
        # Logical variable in Prolog
        X = Term("X")  

        query_term = Term(
            self.function_name, 
            X,
            Constant(label)
        )

        substitution={
            X: Term(
                "tensor", 
                Term(
                    self.dataset_name, 
                    Constant(i)
                )
            )
        }

        q = Query(
            query=query_term, 
            substitution=substitution
        )

        # print("QUERY:", q)

        return q

    
# class MultiStepDataset(Dataset):
#     def __init__(self, indices, labels):
#         self.indices = indices
#         self.labels = labels

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         idx = self.indices[i]
#         label = self.labels[i]

#         return Query(
#             Term("multi_step_attack", Constant(idx)),
#             p=float(label)
#         )