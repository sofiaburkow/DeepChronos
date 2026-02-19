from pathlib import Path
from collections import Counter
import pickle

import numpy as np
import scipy.sparse as sp
import torch

from deepproblog.dataset import Dataset as DPLDataset
from deepproblog.query import Query
from problog.logic import Term, Constant


def load_windowed_data(base_dir: Path, window_size: str, variant: str):
    """
    Load preprocessed windowed data.
    """
    
    dataset_path = base_dir / f"w{window_size}" / variant

    data = {
        split: np.load(dataset_path / f"X_{split}.npy", allow_pickle=True)
        for split in ["train", "test"]
    }

    labels = {
        split: np.load(dataset_path / f"y_{split}_multi_class.npy", allow_pickle=True)
        for split in ["train", "test"]
    }

    return data, labels


class WindowedFlowDataset(torch.utils.data.Dataset):
    """
    Standard PyTorch dataset for windowed flow sequences.
    Returns (X, y).
    """

    def __init__(self, X, y):
        self.X = self._to_dense_tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)

    @staticmethod
    def _to_dense_tensor(X):
        dense = [
            w.toarray() if sp.issparse(w) else np.asarray(w, dtype=np.float32)
            for w in X
        ]
        return torch.tensor(np.stack(dense), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FlowTensorSource(torch.utils.data.Dataset):
    """
    Provides tensors to DeepProbLog neural predicates.
    Returns only X.
    """

    def __init__(self, X):
        self.X = WindowedFlowDataset._to_dense_tensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]

        if isinstance(idx, Constant):
            idx = int(idx.value)

        return self.X[idx]


class FlowDPLDataset(DPLDataset):
    """
    DeepProbLog dataset encoding multi-step attack logic.
    """

    def __init__(
        self,
        labels,
        split_name: str,
        function_name: str,
        lookback_limit: int | None,
        cache_dir: Path,
        cache_id: str,
        save_queries: bool = False,       
        queries_file: Path | None = None,   
    ):
        super().__init__()

        self.labels = labels
        self.split_name = split_name
        self.function_name = function_name
        self.lookback_limit = lookback_limit

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{cache_id}.pkl"

        if self.cache_file.exists():
            self.data = pickle.load(open(self.cache_file, "rb"))
        else:
            self.data = self._build_dataset()
            pickle.dump(self.data, open(self.cache_file, "wb"))

        print("Label distribution:",
              Counter(example[-1] for example in self.data))
        
        self.save_queries = save_queries
        self.queries_file = queries_file

        if self.save_queries:
            self.queries_file.parent.mkdir(parents=True, exist_ok=True)
            self._query_buffer = []

    def _build_dataset(self):
        data = []

        for i in range(len(self.labels)):
            curr_phase = self.labels[i]

            if self.lookback_limit:
                prev = set(
                    self.labels[max(0, i - self.lookback_limit):i]
                )
            else:
                prev = set(
                    self.labels[:i]
                )

            if curr_phase == 0:
                flags = {
                    p: int(p in prev) 
                    for p in range(1, 5)
                }
            else:
                flags = {
                    p: int(p < curr_phase and p in prev) 
                    for p in range(1, 5)
                }

            num_prev = sum(flags.values())
            
            if self.function_name == "multi_step":
                label = (
                    "benign" 
                    if curr_phase == 0 
                    else f"phase{curr_phase}"
                )
            elif self.function_name == "ddos":
                label = (
                    "alarm"
                    if curr_phase == 5 and num_prev == 4
                    else "no_alarm"
                )

            data.append([
                curr_phase,
                flags[1], flags[2], flags[3], flags[4],
                label,
            ])

        return data

    def dump_queries(self):
        if not self.save_queries:
            return

        with open(self.queries_file, "w") as f:
            for q in self._query_buffer:
                f.write(q + "\n")

        print(f"Saved {len(self._query_buffer)} queries to {self.queries_file}")


    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        curr_phase, p1, p2, p3, p4, label = self.data[i]

        X = Term("X")

        sub = {
            X: Term(
                "tensor",
                Term(
                    self.split_name,
                    Constant(i)
                )
            )
        }

        query_term = Term(
            self.function_name,
            X,
            Constant(p1),
            Constant(p2),
            Constant(p3),
            Constant(p4),
            Term(label),
        )

        q = Query(query=query_term, substitution=sub, p=1.0)

        # --- DEBUG STORAGE ---
        if self.save_queries:
            self._query_buffer.append(str(q))

        return q
    

class SubsetDPLDataset(DPLDataset):
    """
    Subset wrapper for DeepProbLog datasets.
    """

    def __init__(self, base_dataset, indices):
        super().__init__()
        self.base = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def to_query(self, i):
        return self.base.to_query(self.indices[i])
    