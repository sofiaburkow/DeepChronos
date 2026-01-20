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


def load_data(resampled_str: str):
    """Load DARPA datasets from disk."""
    datasets_data = {
        "train": np.load(ROOT_DIR / f"processed/{resampled_str}/X_train.npy", allow_pickle=True),
        "test":  np.load(ROOT_DIR / f"processed/{resampled_str}/X_test.npy", allow_pickle=True),
    }
    datasets_labels = {
        "train": np.load(ROOT_DIR / f"processed/{resampled_str}/y_train_multi_class.npy", allow_pickle=True),
        "test":  np.load(ROOT_DIR / f"processed/{resampled_str}/y_test_multi_class.npy", allow_pickle=True),
    }

    return datasets_data, datasets_labels


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

    def __init__(
            self, 
            dataset_name: str, 
            resampled_str: str,
        ):

        self.dataset_name = dataset_name

        datasets_data, datasets_labels = load_data(resampled_str)
        X = datasets_data[dataset_name]

        dense_windows = []
        for _, w in enumerate(X):
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


class DARPADPLDataset(DPLDataset):
    """DeepProbLog dataset for multi-step functions (ddos, multi_step)."""

    def __init__(
            self, 
            dataset_name: str, 
            function_name: str,
            resampled_str: str,
            lookback_limit: int,
            run_id: str
        ):
        super().__init__()

        self.dataset_name = dataset_name
        self.function_name = function_name
        self.resampled_str = resampled_str
        self.lookback_limit = lookback_limit
        self.run_id = run_id

        _, datasets_labels = load_data(self.resampled_str)
        self.labels = datasets_labels[dataset_name]

        self.__set_filenames__()

        print(f"\n--- Preparing {self.function_name} {self.dataset_name} dataset ---")
        print(f"Using {self.resampled_str} data")

        if self.cache_file.exists():
            print(f"Loading cached dataset from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                self.data = pickle.load(f)

            assert len(self.data) == len(self.labels), (
                f"Cached data length ({len(self.data)}) "
                f"!= labels length ({len(self.labels)})"
            )
            
        else:
            print(f"Preparing dataset from scratch...")
            self.data = []
            start = time.time()

            if self.lookback_limit: 
                print(f"Using lookback window of size {self.lookback_limit} for dataset preparation")

                counter = 0
                for i in range(len(self.labels)):
                    curr_phase = self.labels[i]

                    prev_phases = set(self.labels[range(max(0, i - self.lookback_limit), i)]) # exclude current phase
                    phase_flags = {
                        p: 1 if p in prev_phases else 0
                        for p in range(1, 5) # phases 1 to 4
                    }

                    # For analysis
                    num_prev_attack_phases = sum(phase_flags[p] for p in range(1, 5))
                    all_prev_present = (num_prev_attack_phases == 4)
                    if all_prev_present: 
                        counter += 1

                    # Raise alarm if current phase is DDoS and all previous attack phases are present
                    label = "alarm" if curr_phase == 5 and all_prev_present else "no_alarm"
            
                    self.data.append([curr_phase, phase_flags[1], phase_flags[2], phase_flags[3], phase_flags[4], label])
                
                print(f"Number of examples with all previous phases present: {counter}")
            
            else: # full history
                print("Using full history for dataset preparation")

                # Note: this logic assumes that phases appear in order
                num_benign_per_classifier = {p : 0 for p in range(1, 6)}
                history = {p: 0 for p in range(1, 5)} # phases 1 to 4
                for i in range(len(self.labels)):
                    curr_phase = self.labels[i]

                    if curr_phase == 0:
                        # Benign flow, do not update history
                        phase_flags = history.copy()

                        curr_classifier = sum(history.values()) + 1
                        num_benign_per_classifier[curr_classifier] += 1
                    else:
                        # Attack flow, update history
                        if curr_phase < 5:
                            history[curr_phase] = 1

                        phase_flags = {p: 1 if p < curr_phase else 0 for p in range(1, 5)} # phases 1 to 4
                        
                    # Raise alarm if current phase is DDoS and all previous attack phases are present
                    num_prev_phases = sum(phase_flags.values())
                    label = "alarm" if curr_phase == 5 and num_prev_phases == 4 else "no_alarm"
                    
                    self.data.append([curr_phase, phase_flags[1], phase_flags[2], phase_flags[3], phase_flags[4], label])

                print("Number of benign examples seen per classifier during dataset preparation:", num_benign_per_classifier)

            # Final stats
            end = time.time()
            length = end - start
            print(f"Prepared dataset with {len(self.data)} examples in {length:.2f} seconds.")

            # Save for next time
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.data, f)

        counts = Counter(example[-1] for example in self.data)
        print(f"Label distribution:", counts)
    

    def __set_filenames__(self):
        """Set filenames for query and cache files."""
        lookback_limit_str = f"lookback{self.lookback_limit}" if self.lookback_limit else "full_lookback"
        file_name = f"{self.dataset_name}_{self.function_name}_{self.resampled_str}_{lookback_limit_str}"

        QUERIES_DIR = ROOT_DIR / "queries"
        QUERIES_DIR.mkdir(exist_ok=True)
        self.queries_file = QUERIES_DIR / \
            f"{file_name}_{self.run_id}.txt"

        CACHE_DIR = ROOT_DIR / "cache"
        CACHE_DIR.mkdir(exist_ok=True)
        self.cache_file = CACHE_DIR / \
            f"{file_name}.pkl" # no need for run_id in cache file 


    def __len__(self):
        return len(self.labels)


    def to_query(self, i):
        curr_phase, phase_1, phase_2, phase_3, phase_4, label = self.data[i]

        if self.function_name == "ddos":
            outcome = label  # "alarm" or "no_alarm"
        elif self.function_name == "multi_step":
            outcome = "benign" if curr_phase == 0 else f"phase{curr_phase}"  # "benign" or "phasei"

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
            Constant(int(phase_1)),
            Constant(int(phase_2)),
            Constant(int(phase_3)),
            Constant(int(phase_4)),
            Term(outcome)
        )

        q = Query(
            query=query_term, 
            substitution=sub,
            # p=prob
        )

        with open(self.queries_file, "a") as f:
            f.write(f"{q}\n")

        # print("QUERY:", q)

        return q