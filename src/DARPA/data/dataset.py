"""
Dataset wrappers for DARPA flows (windowed inputs) for PyTorch and DeepProbLog.
"""

from pathlib import Path
from collections import Counter
import time
import pickle

import numpy as np
import scipy.sparse as sp
import torch
from deepproblog.dataset import Dataset as DPLDataset
from deepproblog.query import Query
from problog.logic import Term, Constant


# Root directory for dataset files
ROOT_DIR = Path(__file__).parent


# Load datasets once
# data = "original" 
data = "resampled" 

datasets_data = {
    "train": np.load(ROOT_DIR / f"processed/{data}/X_train.npy", allow_pickle=True),
    "test":  np.load(ROOT_DIR / f"processed/{data}/X_test.npy", allow_pickle=True),
}
datasets_labels = {
    "train": np.load(ROOT_DIR / f"processed/{data}/y_train_multi_class.npy", allow_pickle=True),
    "test":  np.load(ROOT_DIR / f"processed/{data}/y_test_multi_class.npy", allow_pickle=True),
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
    """Tensor source for DeepProbLog NNs (returns X only)."""

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
        return len(self.X)

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
    
    
class DARPAReconDataset(DPLDataset):
    def __init__(
            self, 
            dataset_name: str,
            function_name: str,
            run_id: str
        ):
        """
        Dataset of Prolog queries for DeepProbLog.

        :param dataset_name: Dataset to use ("train" or "test")
        :param function_name: Name of Problog function to query
        """

        super().__init__()

        self.dataset_name = dataset_name
        self.function_name = function_name
        self.labels = datasets_labels[dataset_name]
        
        # Debug file for queries
        DEBUG_DIR = ROOT_DIR / "debug"
        DEBUG_DIR.mkdir(exist_ok=True)
        self.debug_file = DEBUG_DIR / f"{self.function_name}_queries_{run_id}.txt"

        # Prepare multi-step data
        CACHE_DIR = ROOT_DIR / "cache"
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"{self.dataset_name}_{self.function_name}.pkl"

        if cache_file.exists():
            print(f"Loading cached {self.dataset_name} dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            print(f"Preparing {self.function_name} {self.dataset_name} dataset from scratch...")
        
            self.data = []

            for label in self.labels:
                if is_recon(int(label)):
                    self.data.append("alarm")
                else:
                    self.data.append("no_alarm")
    
        # Print label distribution
        counts = Counter(self.data)
        print(f"Label distribution ({function_name}):", counts)

    def __len__(self):
        return len(self.labels)

    def to_query(self, i):
        """Return a Query object for the i-th example."""

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

        q = Query(
            query=query_term, 
            substitution=sub
        )

        # print("QUERY:", q)

        return q


class DARPAMultiStepDataset(DPLDataset):
    def __init__(
            self, 
            dataset_name: str, 
            function_name: str, 
            run_id: str
        ):
        super().__init__()

        self.dataset_name = dataset_name
        self.function_name = function_name
        self.labels = datasets_labels[dataset_name]
        
        # Debug file for queries
        DEBUG_DIR = ROOT_DIR / "debug"
        DEBUG_DIR.mkdir(exist_ok=True)
        self.debug_file = DEBUG_DIR / f"{self.function_name}_queries_{run_id}.txt"

        # Prepare multi-step data
        CACHE_DIR = ROOT_DIR / "cache"
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"{self.dataset_name}_{self.function_name}.pkl"

        if cache_file.exists():
            print(f"Loading cached {self.dataset_name} dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            print(f"Preparing {self.function_name} {self.dataset_name} dataset from scratch...")
            self.data = []
            start = time.time()

            if self.function_name == "ddos":
                DELTA = 20000 
                print(f"Using lookback window size DELTA={DELTA}")

                counter = 0
                for i in range(len(self.labels)):
                    curr_phase = self.labels[i]

                    prev_phases = set(self.labels[range(max(0, i - DELTA), i)]) # exclude current phase
                    phase_flags = {
                        p: 1 if p in prev_phases else 0
                        for p in range(1, 5) # phases 1 to 4
                    }

                    phase_1 = phase_flags[1]
                    phase_2 = phase_flags[2]
                    phase_3 = phase_flags[3]
                    phase_4 = phase_flags[4]

                    # For analysis
                    num_prev_attack_phases = sum(phase_flags[p] for p in range(1, 5))
                    all_prev_present = (num_prev_attack_phases == 4)
                    if all_prev_present: 
                        counter += 1

                    label = "alarm" if is_ddos(curr_phase) and all_prev_present else "no_alarm"
                    
                    self.data.append([curr_phase, phase_1, phase_2, phase_3, phase_4, label])
                
                print(f"Number of examples with all previous phases present: {counter}")
            
            elif self.function_name == "multi_step":
                for i in range(len(self.labels)):
                    curr_phase = self.labels[i]
                    
                    if curr_phase > 0 and curr_phase < 5:
                        print(curr_phase)

                    prev_phases_flags = [1 if p < curr_phase else 0 for p in range(1, 5)]
                    phase_1, phase_2, phase_3, phase_4 = prev_phases_flags

                    all_prev_present = sum(prev_phases_flags) == 4
                    label = "alarm" if is_ddos(curr_phase) and all_prev_present else "no_alarm"
                    
                    self.data.append([curr_phase, phase_1, phase_2, phase_3, phase_4, label])

            # Final stats
            end = time.time()
            length = end - start
            print(f"Prepared {self.function_name} {self.dataset_name} dataset with {len(self.data)} examples in {length:.2f} seconds.")

            # Save for next time
            with open(cache_file, "wb") as f:
                pickle.dump(self.data, f)

        counts = Counter(example[-1] for example in self.data)
        print(f"Label distribution ({self.function_name}):", counts)

    def __len__(self):
        return len(self.labels)

    def to_query(self, i):
        curr_phase, phase_1, phase_2, phase_3, phase_4, label = self.data[i]

        # prob = float(label == "alarm")

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

        if self.function_name == "ddos":
            # DDOS supervision
            query_term = Term(
                self.function_name,
                X,
                Constant(int(phase_1)),
                Constant(int(phase_2)),
                Constant(int(phase_3)),
                Constant(int(phase_4)),
                Term(label) # "alarm" or "no_alarm"
            )
        elif self.function_name == "multi_step":
            l = "benign" if label == "no_alarm" else f"phase{curr_phase}"
            # Multi-step supervision
            query_term = Term(
                self.function_name,
                X,
                Constant(int(phase_1)),
                Constant(int(phase_2)),
                Constant(int(phase_3)),
                Constant(int(phase_4)),
                Term(l) # "benign" or "phasei"
            )

        q = Query(
            query=query_term, 
            substitution=sub,
            # p=prob
        )

        with open(self.debug_file, "a") as f:
            f.write(f"{q}\n")

        # print("QUERY:", q)

        return q


# Helper functions when creating logical queries for DARPA

def is_recon(phase: int) -> bool:
    return phase == 1 or phase == 2

def is_ddos(label: int) -> bool:
    return label == 5
