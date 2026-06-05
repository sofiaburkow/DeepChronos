from pathlib import Path
from collections import Counter, defaultdict
import pickle
import copy

import numpy as np
import scipy.sparse as sp
import torch

from deepproblog.dataset import Dataset as DPLDataset
from deepproblog.query import Query
from problog.logic import Term, Constant

from src.feature_engineering.features import DPL_FEATURES


def load_windowed_data(data_dir: Path, subset: str):
    """
    Load preprocessed windowed data.
    """

    data = {
        split: np.load(data_dir / f"X_{split}.npy", allow_pickle=True)
        for split in ["train", "test"]
    }

    labels = {
        split: np.load(data_dir / f"y_{split}.npy", allow_pickle=True)
        for split in ["train", "test"]
    }

    dpl_features = {
        split: {
            key: np.load(data_dir / f"{key}_{split}.npy", allow_pickle=True)
            for key in DPL_FEATURES
        }
        for split in ["train", "test"]
    }

    if subset == "balanced":
        y_train = labels["train"]
        label_counts = np.bincount(y_train)
        attack_classes = np.arange(1, len(label_counts))
        max_attack_count = label_counts[attack_classes].max()
        subset = f"{max_attack_count}b{max_attack_count}a"

    if subset != "full":
        subset_indices = np.load(data_dir / f"subsets/train_{subset}.npy")
        data["train"] = data["train"][subset_indices]
        labels["train"] = labels["train"][subset_indices]
        dpl_features["train"] = {key: value[subset_indices] for key, value in dpl_features["train"].items()}
    
    return data, labels, dpl_features


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
        logic_features: dict[str, np.ndarray],
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

        # Set up cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{cache_id}.pkl"

        # if self.cache_file.exists():
        #     self.data = pickle.load(open(self.cache_file, "rb"))
        # else:
        # For now, always rebuild dataset
        self.dataset_name = logic_file.split("_")[0]
        if self.dataset_name == "darpa":
            data = self._build_dataset_darpa()
        elif self.dataset_name == "ait":
            data = self._build_dataset_ait()
        
        print("Label distribution:",
              Counter(example["label"] for example in data))
        flag_combos = [
            tuple(example["flags"].values())
            for example in data
        ]
        print("Phase‑flag combinations:", Counter(flag_combos))

        if "miss_flags" in logic_file:
            data = self._corrupt_data(data=data, phase_to_corrupt=2, corruption_rate=0.3, seed=123)

            flag_combos = [
                tuple(example["flags"].values())
                for example in data
            ]
            print("Phase‑flag combinations after corruption:", Counter(flag_combos))
            
        self.data = data
        pickle.dump(self.data, open(self.cache_file, "wb"))
        
        self.save_queries = save_queries
        self.queries_file = queries_file

        if self.save_queries:
            self.queries_file.parent.mkdir(parents=True, exist_ok=True)
            self._query_buffer = []
    

    def __len__(self):
        return len(self.data)


    def _build_dataset_ait(self):
        print("Building AIT-LDS V2 dataset with multi-step logic...")
        data = []

        # AIT specific
        num_attack_phases = 4
        flag_map_ait = {
            "phase1": [0,0,0], 
            "phase2": [1,0,0],
            "phase3": [1,1,0],
            "phase4": [1,1,1],
        }

        # Running history
        attacker_phase_history = defaultdict(set)  # attacker_ip -> set of phases seen so far

        # Sort temporally
        times = self.logic_features["start_time"].astype(float)
        time_order = np.argsort(times)

        for sorted_i in time_order:

            # Load logic features
            src_ip = self.logic_features["src_ip"][sorted_i]
            dst_ip = self.logic_features["dst_ip"][sorted_i]
            dport = self.logic_features["dport"][sorted_i]

            prot_map = {"icmp": 1, "tcp": 6, "udp": 17}
            protocol = str(self.logic_features["proto"][sorted_i])
            protocol = prot_map.get(protocol, 0) # default to 0 if unknown

            local_orig = str(self.logic_features["local_orig"][sorted_i]) # T or F
            local_resp = str(self.logic_features["local_resp"][sorted_i]) # T or F
            local_orig = 1 if local_orig == "T" else 0
            local_resp = 1 if local_resp == "T" else 0

            # Determine label
            curr_phase = self.labels[sorted_i]
            if curr_phase == 0:
                label = "benign"
            else:
                label = f"phase{curr_phase}"

            # Phase flags based on history
            src_history = attacker_phase_history[src_ip]
            dst_history = attacker_phase_history[dst_ip]
            history = src_history.union(dst_history)

            if curr_phase == 0:
                flags = {
                    p: int(p in history)
                    for p in range(1, num_attack_phases)
                }
            else:
                flags = {
                    p: 1 if p < curr_phase else 0
                    for p in range(1, num_attack_phases)
                }
            
            # Sanity check
            if curr_phase != 0:
                expected_flags = flag_map_ait.get(label, [0,0,0])
                curr_flags = [flags.get(p, 0) for p in range(1, num_attack_phases)]
                if curr_flags != expected_flags:
                    print(f"Sanity check failed for flow with label {label}: expected {expected_flags}, got {curr_flags}")
                    flags = [f for f in expected_flags.values()]

            # Store data
            data.append({
                "dpl_index": int(sorted_i),
                "orig_index": int(self.logic_features["orig_index"][sorted_i]),
                "phase": int(curr_phase),
                "label": label,
                "flags": flags,
                "local_orig": local_orig,
                "local_resp": local_resp,
                "dport": dport,
                "protocol": protocol,
            })

            # Update history
            if curr_phase != 0:
                attacker_phase_history[src_ip].add(curr_phase)

            if curr_phase == 2:
                attacker_phase_history[src_ip].add(1)

        # Restore original shuffled order
        data_sorted = data
        data = [None] * len(data_sorted)
        for new_pos, original_pos in enumerate(time_order):
            data[original_pos] = data_sorted[new_pos]

        return data


    def _build_dataset_darpa(self):
        print("Building DARPA 2000 dataset with multi-step logic...")
        data = []

        # DARPA specific
        num_attack_phases = 5
        flag_map_darpa = {
            "phase1": [0,0,0,0], 
            "phase2": [1,0,0,0],
            "phase3": [1,1,0,0],
            "phase4": [1,1,1,0],
            "phase5": [1,1,1,1],
        }

        # Running history
        attacker_phase_history = defaultdict(set)  # attacker_ip -> set of phases seen so far
        compromised_flag = False

        # Sort temporally
        times = self.logic_features["start_time"].astype(float)
        time_order = np.argsort(times)

        for sorted_i in time_order:
            # Load logic features
            src_ip = self.logic_features["src_ip"][sorted_i]
            dst_ip = self.logic_features["dst_ip"][sorted_i]
            dport = self.logic_features["dport"][sorted_i]

            prot_map = {"icmp": 1, "tcp": 6, "udp": 17}
            protocol = str(self.logic_features["proto"][sorted_i])
            protocol = prot_map.get(protocol, 0) # default to 0 if unknown

            local_orig = str(self.logic_features["local_orig"][sorted_i]) # T or F
            local_resp = str(self.logic_features["local_resp"][sorted_i]) # T or F
            local_orig = 1 if local_orig == "T" else 0
            local_resp = 1 if local_resp == "T" else 0

            # Determine label
            curr_phase = self.labels[sorted_i]
            if curr_phase == 0:
                label = "benign"
            else:
                label = f"phase{curr_phase}"

            # Phase flags based on history
            src_history = attacker_phase_history[src_ip]
            dst_history = attacker_phase_history[dst_ip]
            history = src_history.union(dst_history)

            flags = {
                p: int(p in history)
                for p in range(1, num_attack_phases) # flags for phases 1 to 4
            }

            # Attack phases should not have their own flag set
            if curr_phase != 0 and curr_phase != num_attack_phases: 
                flags[curr_phase] = 0
            
            # If already compromised, and not phase 4, set all flags
            if compromised_flag and curr_phase != (num_attack_phases - 1): 
                flags = {p: 1 for p in range(1, num_attack_phases)} 

            # Sanity check
            if curr_phase != 0:
                expected_flags = flag_map_darpa.get(label, [0,0,0,0])
                curr_flags = [flags.get(p, 0) for p in range(1, num_attack_phases)]
                if curr_flags != expected_flags:
                    print(f"Sanity check failed for flow with label {label}: expected {expected_flags}, got {flags}")

            # Store data
            data.append({
                "dpl_index": int(sorted_i),
                "orig_index": int(self.logic_features["orig_index"][sorted_i]),
                "phase": int(curr_phase),
                "label": label,
                "flags": flags,
                "local_orig": local_orig,
                "local_resp": local_resp,
                "dport": dport,
                "protocol": protocol,
            })

            # Update history
            attacker_phase_history[src_ip].add(curr_phase)

            if curr_phase == (num_attack_phases - 1):  # phase4
                compromised_flag = True

        # Restore original shuffled order
        data_sorted = data
        data = [None] * len(data_sorted)
        for new_pos, original_pos in enumerate(time_order):
            data[original_pos] = data_sorted[new_pos]

        return data
    
    
    def _corrupt_data(self, data, phase_to_corrupt, corruption_rate, seed):
        data_corrupted = copy.deepcopy(data)
        
        rng = np.random.default_rng(seed)
        phase_idx = [i for i, rec in enumerate(data) if rec["phase"] == phase_to_corrupt]
        indexes_to_corrupt = rng.choice(phase_idx, size=int(corruption_rate * len(phase_idx)), replace=False)
    
        for i in indexes_to_corrupt:
            for p in list(data_corrupted[i]["flags"].keys()):
                data_corrupted[i]["flags"][p] = 0

        return data_corrupted
    

    def dump_queries(self):
        if not self.save_queries:
            return

        with open(self.queries_file, "w") as f:
            for q in self._query_buffer:
                f.write(q + "\n")

        print(f"Saved {len(self._query_buffer)} queries to {self.queries_file}")


    def to_query(self, i):  

        ex = self.data[i]
        flags = ex["flags"]

        X = Term("X")

        sub = {
            X: Term("tensor", Term(self.split_name, Constant(i)))
        }

        if self.dataset_name == "darpa":
            query_term = Term(
                "multi_step",
                X,
                Constant(flags.get(1, 0)),
                Constant(flags.get(2, 0)),
                Constant(flags.get(3, 0)),
                Constant(flags.get(4, 0)),
                Constant(ex["local_orig"]),
                Constant(ex["local_resp"]),
                Constant(ex["dport"]),
                Constant(ex["protocol"]),
                Term(ex["label"]),
            )
        elif self.dataset_name == "ait":
            query_term = Term(
                "multi_step",
                X,
                Constant(flags.get(1, 0)),
                Constant(flags.get(2, 0)),
                Constant(flags.get(3, 0)),
                Constant(ex["local_orig"]),
                Constant(ex["local_resp"]),
                Constant(ex["dport"]),
                Constant(ex["protocol"]),
                Term(ex["label"]),
            )

        q = Query(query=query_term, substitution=sub, p=1.0)

        if self.save_queries:
            self._query_buffer.append(str(q))

        return q