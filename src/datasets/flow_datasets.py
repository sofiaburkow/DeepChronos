from pathlib import Path
from collections import Counter, defaultdict, deque
import pickle

import numpy as np
import scipy.sparse as sp
import torch

from deepproblog.dataset import Dataset as DPLDataset
from deepproblog.query import Query
from problog.logic import Term, Constant

from src.feature_engineering.features import FEATURES


def load_windowed_data(data_dir: Path, fraction: int):
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

    logic_features = {
        split: {
            key: np.load(data_dir / f"{key}_{split}.npy", allow_pickle=True)
            for key in FEATURES.logic_features
        }
        for split in ["train", "test"]
    }

    metadata_features = {
        split: {
            key: np.load(data_dir / f"{key}_{split}.npy", allow_pickle=True)
            for key in FEATURES.metadata_features
        }
        for split in ["train", "test"]
    }

    if fraction < 100:
        subset_indices = np.load(data_dir / f"subsets/train_{fraction}.npy")
        data["train"] = data["train"][subset_indices]
        labels["train"] = labels["train"][subset_indices]
        logic_features["train"] = {key: value[subset_indices] for key, value in logic_features["train"].items()}
        metadata_features["train"] = {key: value[subset_indices] for key, value in metadata_features["train"].items()}

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

        scenario = logic_file.split("_")[0]
        if scenario == "darpa":
            self.num_phases = 5
        elif scenario == "ait":
            self.num_phases = 4
        else:
            raise ValueError(f"Unknown scenario in logic file: {logic_file}")

        self.logic_features = logic_features
        self.metadata_features = metadata_features

        # Set up cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{cache_id}.pkl"

        # if self.cache_file.exists():
        #     self.data = pickle.load(open(self.cache_file, "rb"))
        # else:
        # For now, always rebuild dataset
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
        # phase_counts = Counter()
        seen_phases = set()

        time_window = 20  # seconds
        recent_sources = defaultdict(lambda: deque())
        source_counts = defaultdict(Counter)

        recent_dst = defaultdict(lambda: deque())
        dst_counts = defaultdict(Counter)

        times = self.metadata_features["start_time"].astype(float)

        # Sort temporally
        time_order = np.argsort(times)

        # Reverse mapping to restore order later
        inverse_order = np.argsort(time_order)

        # Time sanity check
        last_time = 0  
        # for i, curr_phase in enumerate(self.labels):
        for sorted_i in time_order:

            # Time sanity check
            time_i = times[sorted_i]
            if time_i < last_time:
                print("Warning: timestamps are not sorted!")
                break
            last_time = time_i  

            curr_phase = self.labels[sorted_i]

            label = "benign" if curr_phase == 0 else f"phase{curr_phase}"

            # Flags
            if curr_phase == 0:
                flags = {
                    p: int(p in seen_phases) 
                    for p in range(1, self.num_phases)
                }
            else:
                flags = {
                    p: int(p < curr_phase and p in seen_phases) 
                    for p in range(1, self.num_phases)
                }

            next_phase = sum(flags.values()) + 1
            
            # Local orig/resp
            local_orig = str(self.logic_features["local_orig"][sorted_i]) # T or F
            local_orig = 1 if local_orig == "T" else 0
            local_resp = str(self.logic_features["local_resp"][sorted_i]) # T or F
            local_resp = 1 if local_resp == "T" else 0

            # Protocol
            prot_map = {"icmp": 1, "tcp": 6, "udp": 17}
            protocol = str(self.logic_features["proto"][sorted_i])
            protocol = prot_map.get(protocol, 0) # default to 0 if unknown

            # Destination port
            dport = self.logic_features["dport"][sorted_i]
            # DDoS metrics
            curr_time = self.metadata_features["start_time"][sorted_i]
            src_ip = self.logic_features["src_ip"][sorted_i]
            dst_ip = self.logic_features["dst_ip"][sorted_i]
            in_queue = recent_sources[dst_ip]
            in_counts = source_counts[dst_ip]

            # remove old
            while in_queue and curr_time - in_queue[0][0] > time_window:
                _, old_src = in_queue.popleft()
                in_counts[old_src] -= 1
                if in_counts[old_src] == 0:
                    del in_counts[old_src]

            in_queue.append((curr_time, src_ip))
            in_counts[src_ip] += 1

            ddos_count = len(in_queue)
            unique_sources = len(in_counts)
            ddos_rate = ddos_count / time_window

            out_queue = recent_dst[src_ip]
            out_counts = dst_counts[src_ip]

            while out_queue and curr_time - out_queue[0][0] > time_window:
                _, old_dst = out_queue.popleft()
                out_counts[old_dst] -= 1
                if out_counts[old_dst] == 0:
                    del out_counts[old_dst]

            out_queue.append((curr_time, dst_ip))
            out_counts[dst_ip] += 1

            fanout_count = len(out_queue)
            unique_targets = len(out_counts)
            fanout_rate = fanout_count / time_window

            # Store data
            data.append({
                "dpl_index": int(sorted_i),
                "orig_index": int(self.metadata_features["orig_index"][sorted_i]),
                "phase": int(curr_phase),
                # "phase_counts": dict(phase_counts),
                "flags": flags,
                "next_phase": next_phase,
                "local_orig": local_orig,
                "local_resp": local_resp,
                "dport": dport,
                "protocol": protocol,
                "ddos_count": ddos_count,
                "ddos_rate": ddos_rate,
                "unique_sources": unique_sources,
                "fanout_count": fanout_count,
                "fanout_rate": fanout_rate,
                "unique_targets": unique_targets,
                "label": label,
            })

            # update history
            # phase_counts[curr_phase] += 1
            seen_phases.add(curr_phase)
        
        # Restore original shuffled order
        data_sorted = data
        data = [None] * len(data_sorted)

        for new_pos, original_pos in enumerate(time_order):
            data[original_pos] = data_sorted[new_pos]

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
        next_phase = example["next_phase"]
        local_orig = example["local_orig"]
        local_resp = example["local_resp"]
        dport = example["dport"]
        protocol = example["protocol"]

        ddos_rate = example["ddos_rate"]
        unique_sources = example["unique_sources"]

        fanout_rate = example["fanout_rate"]
        unique_targets = example["unique_targets"]

        label = example["label"]

        ddos_rate_signal = 1 if ddos_rate > 5.0 else 0

        if unique_sources > 10 or unique_targets > 10:
            ddos_signal = 1
        else:
            ddos_signal = 0 

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
            Constant(next_phase),
            X,
            Constant(local_orig),
            Constant(local_resp),
            Constant(dport),
            Constant(protocol), 
            Constant(ddos_rate_signal), # not used in logic for now
            Constant(ddos_signal),
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