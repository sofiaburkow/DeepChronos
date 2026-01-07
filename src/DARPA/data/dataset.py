"""
Dataset wrappers for DARPA flows (windowed inputs) for PyTorch and DeepProbLog.
"""

from pathlib import Path
from typing import Tuple, List

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
datasets_labels = {
    "train": np.load(ROOT_DIR / "processed/y_train.npy", allow_pickle=True),
    "test":  np.load(ROOT_DIR / "processed/y_test.npy", allow_pickle=True),
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

        self.dataset_name = dataset_name
        self.function_name = function_name
        self.phases = datasets_labels[dataset_name]

        self.labels = []
        self.indices = []
        self.num_indices = 10  # for multi-step detection

        self.mode = "multi_step"
        # self.mode = "recon"
        print(f"Creating DARPADPLDataset in mode: {self.mode}")

        if self.mode == "recon":
            for phase in self.phases:
                if is_recon(int(phase)):
                    self.labels.append("alarm")
                else:
                    self.labels.append("no_alarm")

        elif self.mode == "multi_step":
            for i in range(len(self.phases) - self.num_indices):
                indices = list(range(i, i + self.num_indices))
                self.indices.append(indices)

                labels = [int(phase) for phase in self.phases[i : i + self.num_indices]]
                
                if is_multi_step(labels):
                    # Debug statement
                    print("Actually a multi-step attack detected in indices", indices, "with phases", labels)
                    self.labels.append("alarm")
                else:
                    self.labels.append("no_alarm")

    def __len__(self):
        return len(self.labels)

    def to_query(self, i):
        """Return a Query object for the i-th example."""

        if self.mode == "recon":
            # Logical variable in Prolog
            X = Term("X")  
            
            sub={
                X: Term(
                    "tensor", 
                    Term(
                        self.dataset_name, 
                        Constant(i)
                    )
                )
            }

            query_term = Term(
                self.function_name, 
                X,
                Term(self.labels[i])
            )

        elif self.mode == "multi_step":
            
            indices = self.indices[i]
            sub_flows = [Term("window_{}".format(x)) for x in range(len(indices))]
            flows = [
                Term(
                    "tensor", 
                    Term(
                        self.dataset_name, 
                        Constant(idx)
                    )
                )
                for idx in indices
            ]
            sub = {sub_flows[j]: flows[j] for j in range(len(flows))}

            query_term = Term(
                self.function_name, 
                *sub_flows,
                Term(self.labels[i])
            )

        q = Query(
            query=query_term, 
            substitution=sub
        )

        # print("QUERY:", q)

        return q
    

def is_recon(phase: int) -> bool:
    return phase == 1 or phase == 2

def is_multi_step(phases: List[int]) -> bool:
    attack_phases = list(set(phases))
    return len(attack_phases) >= 2