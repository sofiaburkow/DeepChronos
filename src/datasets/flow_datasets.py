from pathlib import Path
from collections import Counter
import pickle

import numpy as np
import scipy.sparse as sp
import torch

from deepproblog.dataset import Dataset as DPLDataset
from deepproblog.query import Query
from problog.logic import Term, Constant

from src.feature_engineering.features import FEATURES


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
        split: np.load(dataset_path / f"y_{split}.npy", allow_pickle=True)
        for split in ["train", "test"]
    }

    logic_features = {
        split: {
            key: np.load(dataset_path / f"{key}_{split}.npy", allow_pickle=True)
            for key in FEATURES.logic_features
        }
        for split in ["train", "test"]
    }

    metadata_features = {
        split: {
            key: np.load(dataset_path / f"{key}_{split}.npy", allow_pickle=True)
            for key in FEATURES.metadata_features
        }
        for split in ["train", "test"]
    }

    return data, labels, logic_features, metadata_features


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
        labels: np.ndarray,
        logic_features: dict,
        metadata_features: dict,
        split_name: str,
        logic_file: str,
        cache_dir: Path,
        cache_id: str,
        save_queries: bool = False,       
        queries_file: Path | None = None,   
    ):
        super().__init__()

        print(f"\n Initializing {split_name} dataset...")

        self.labels = labels
        self.split_name = split_name
        self.logic_file = logic_file

        self.logic_features = logic_features
        self.metadata_features = metadata_features

        # Set up cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{cache_id}.pkl"

        if self.cache_file.exists():
            self.data = pickle.load(open(self.cache_file, "rb"))
        else:
            self.data = self._build_dataset()
            pickle.dump(self.data, open(self.cache_file, "wb"))

        print("Label distribution:",
              Counter(example["label"] for example in self.data))
        
        print("Phase flags distribution:",
              Counter(sum(example["flags"].values()) for example in self.data))
        
        self.save_queries = save_queries
        self.queries_file = queries_file

        if self.save_queries:
            self.queries_file.parent.mkdir(parents=True, exist_ok=True)
            self._query_buffer = []
    

    def __len__(self):
        return len(self.data)


    def _build_dataset(self):
        data = []

        # Running history
        phase_counts = Counter()
        seen_phases = set()

        for i, curr_phase in enumerate(self.labels):

            # Flags
            if curr_phase == 0:
                flags = {
                    p: int(p in seen_phases) 
                    for p in range(1, 5)
                }
            else:
                flags = {
                    p: int(p < curr_phase and p in seen_phases) 
                    for p in range(1, 5)
                }

            label = "benign" if curr_phase == 0 else f"phase{curr_phase}"

            # store data
            data.append({
                "dpl_index": i,
                "orig_index": int(self.metadata_features["orig_index"][i]),
                "phase": int(curr_phase),
                "phase_counts": dict(phase_counts),
                "flags": flags,
                "label": label,
            })

            # update history
            phase_counts[curr_phase] += 1
            seen_phases.add(curr_phase)

        return data
    

    def dump_queries(self):
        if not self.save_queries:
            return

        with open(self.queries_file, "w") as f:
            for q in self._query_buffer:
                f.write(q + "\n")

        print(f"Saved {len(self._query_buffer)} queries to {self.queries_file}")


    def to_query(self, i):

        example = self.data[i]
        
        next = sum(example["flags"].values()) + 1

        phase = example["phase"]
        curr_count = int(example["phase_counts"].get(phase, 0))
        count_bin = min(curr_count, 2) # cap count at 3

        label = example["label"]

        # src_ip = self.logic_features["src_ip"][i]
        # dst_ip = self.logic_features["dst_ip"][i]
        # sport = self.logic_features["sport"][i]
        dport = self.logic_features["dport"][i]

        protocol = str(self.logic_features["proto"][i])
        prot_map = {"icmp": 1, "tcp": 6, "udp": 17}
        protocol = prot_map.get(protocol, 0) # default to 0 if unknown

        # service = self.logic_features["service"][i]
        # print(f"Example {i} service: {service}")

        local_orig = str(self.logic_features["local_orig"][i]) # T or F
        local_orig = 1 if local_orig == "T" else 0
        local_resp = str(self.logic_features["local_resp"][i]) # T or F
        local_resp = 1 if local_resp == "T" else 0

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
            "multi_step",
            X,
            Constant(next),
            Constant(count_bin),
            Constant(local_orig),
            Constant(local_resp),
            Constant(dport),
            Constant(protocol),
            Term(label),
        )

        q = Query(
            query=query_term, 
            substitution=sub, 
            p=1.0
        )

        # --- DEBUG STORAGE ---
        if self.save_queries:
            self._query_buffer.append(str(q))

        # print(f"Query {i}: {q}")

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