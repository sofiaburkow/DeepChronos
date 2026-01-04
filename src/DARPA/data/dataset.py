import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset
from deepproblog.dataset import Dataset as DPLDataset

from deepproblog.query import Query
from problog.logic import Term, Constant


# Load datasets
ROOT_DIR = Path(__file__).parent

datasets_data = {
    "train": np.load(ROOT_DIR / "processed/X_train.npy", allow_pickle=True),
    "test":  np.load(ROOT_DIR / "processed/X_test.npy", allow_pickle=True),
}
# datasets_labels = {
#     "train": np.load(ROOT_DIR / "processed/y_train.npy", allow_pickle=True),
#     "test":  np.load(ROOT_DIR / "processed/y_test.npy", allow_pickle=True),
# }
datasets_labels = {
    "train": np.load(ROOT_DIR / "processed/y_phase_1_train.npy", allow_pickle=True),
    "test":  np.load(ROOT_DIR / "processed/y_phase_1_test.npy", allow_pickle=True),
}


class DARPAWindowed(TorchDataset):
    def __init__(
            self, 
            dataset_name: str,
        ):
        """ 
        Generic DARPA dataset for PyTorch using windowed data.
        
        :param dataset_name: Dataset to use ("train" or "test")
        """

        self.data = datasets_data[dataset_name]
        self.labels = datasets_labels[dataset_name]

        dense_windows = []
        for i, w in enumerate(self.data):
            if sp.issparse(w):
                dense_windows.append(w.toarray())
            else:
                dense_windows.append(np.asarray(w, dtype=np.float32))

        # Stack into (N, seq_len, n_features)
        X_dense = np.stack(dense_windows).astype(np.float32)

        self.X = torch.from_numpy(X_dense)
        self.y = torch.from_numpy(np.asarray(self.labels, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # print(f"[DARPAWindowed] Fetching index: {idx}")

        # Handle (Constant(i),)
        if isinstance(idx, tuple):
            idx = idx[0]

        # Handle Constant(i)
        if isinstance(idx, Constant):
            idx = int(idx.value)

        tensor = self.X[idx]
        # print(f"[DARPAWindowed] Tensor shape: {tensor.shape}")

        return tensor
    

class DARPAOperator(DPLDataset):
    def __init__(
            self, 
            dataset_name: str,
            function_name: str,
        ):
        """
        Generic DARPA dataset for DeepProbLog.
        
        :param dataset_name: Dataset to use ("train" or "test")
        :param function_name: Name of Problog function to query
        """

        super(DARPAOperator, self).__init__()
        
        assert dataset_name in ["train", "test"]
        self.dataset_name = dataset_name
        self.y = datasets_labels[self.dataset_name]
        self.function_name = function_name

        # Sanity check
        print(f"[DARPAOperator] Loaded {len(self.y)} samples for dataset '{dataset_name}'")

    def __len__(self):
        return len(self.y)

    def to_query(self, i):

        expected_result = int(self.y[i])
        X = Term("X")  # logical variable
        q = Query(
                Term(
                    self.function_name, 
                    X, 
                    Constant(expected_result)
                ),
                substitution={
                    X: Term(
                        "tensor", 
                        Term(
                            self.dataset_name, 
                            Constant(i)
                        )
                    )
                }
            )

        if q is None:
            raise RuntimeError("to_query returned None")
        
        print("QUERY:", q)

        return q
