import numpy as np
import torch
import scipy.sparse as sp

from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    """
    PyTorch dataset for LSTM training.

    Handles windowed inputs where each sample may be a sparse matrix.
    """

    def __init__(self, X, y):
        """
        X: array-like of shape (N,), each element is
           - scipy.sparse matrix of shape (seq_len, n_features), or
           - dense ndarray of same shape
        y: array-like of shape (N,)
        """

        dense_windows = []

        for i, w in enumerate(X):
            if sp.issparse(w):
                dense_windows.append(w.toarray())
            else:
                dense_windows.append(np.asarray(w, dtype=np.float32))

        # Stack into (N, seq_len, n_features)
        X_dense = np.stack(dense_windows).astype(np.float32)

        self.X = torch.from_numpy(X_dense)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]