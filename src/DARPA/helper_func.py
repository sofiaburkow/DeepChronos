import numpy as np
from scipy.sparse import load_npz

import torch
from collections import Counter 

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import numpy as np


def evaluate(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            probs = model(X_batch)
            preds = (probs >= 0.5).long()

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return acc, precision, recall, f1, cm, y_pred


def misclassified_samples(y_true, y_pred, y_true_phases):

    # Identify misclassified samples and count them per phase
    misclassified_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    misclassified_phases = [y_true_phases[i] for i in misclassified_indices]
    counts = Counter(misclassified_phases)

    num_phases = 6  # phases 0 = benign, 1-5 = attack phases
    phases = list(range(num_phases))
    num_misclassified = [int(counts.get(p, 0)) for p in phases]

    print(f"Number of misclassified samples: {len(misclassified_indices)}")
    print("Misclassified samples per phase:")
    for phase, count in zip(phases, num_misclassified):
        print(f"Phase {phase}: {count} samples")
    print()

    # Return as dictionary
    return {
        "total_misclassified": len(misclassified_indices),
        "per_phase": {phase: int(counts.get(phase, 0)) for phase in phases}
    }


def load_datasets(dataset_dir, phase, sparse=False):
    if sparse:
        X_train = load_npz(f"{dataset_dir}/X_train.npz")
        X_test = load_npz(f"{dataset_dir}/X_test.npz")
    else:
        X_train = np.load(f"{dataset_dir}/X_train.npy", allow_pickle=True)
        X_test = np.load(f"{dataset_dir}/X_test.npy", allow_pickle=True)

    y_train = np.load(f"{dataset_dir}/y_phase_{phase}_train.npy", allow_pickle=True)
    y_test = np.load(f"{dataset_dir}/y_phase_{phase}_test.npy", allow_pickle=True) 

    y_phases_train = np.load(f"{dataset_dir}/y_train.npy", allow_pickle=True)
    y_phases_test = np.load(f"{dataset_dir}/y_test.npy", allow_pickle=True)
    
    return X_train, y_train, y_phases_train, X_test, y_test, y_phases_test