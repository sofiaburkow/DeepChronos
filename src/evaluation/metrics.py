import json
from collections import Counter

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate(
    model,
    dataloader,
    multi_class: bool = False,
    average: str = "auto",
    device: torch.device | None = None,
):
    """
    Evaluate a PyTorch model on a given dataloader.
    """

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            # ---- Prediction handling ----
            if multi_class:
                # multi-label -> single class via argmax
                probs = torch.sigmoid(outputs)
                preds = probs.argmax(dim=1)
            else:
                # supports logits or probabilities
                if outputs.ndim == 1 or outputs.shape[1] == 1:
                    preds = (outputs > 0).long().view(-1)
                else:
                    preds = outputs.argmax(dim=1)

            # ---- Ground truth handling ----
            if y_batch.ndim > 1:
                y_batch = y_batch.argmax(dim=1)

            y_true.append(y_batch.cpu())
            y_pred.append(preds.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    acc = accuracy_score(y_true, y_pred)

    # ---- Averaging strategy ----
    labels = np.unique(y_true)
    if average == "auto":
        avg = "binary" if len(labels) == 2 else "macro"
    else:
        avg = average

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=avg,
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return acc, precision, recall, f1, cm, y_pred



def misclassified_samples(y_true, y_pred, y_true_phases):
    """
    Analyze misclassified samples and count them per attack phase.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_true_phases: True attack phases corresponding to each sample
    """
    misclassified_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    misclassified_phases = [y_true_phases[i] for i in misclassified_indices]
    counts = Counter(misclassified_phases)

    num_phases = 6  # phases 0 = benign, 1-5 = attack phases
    phases = list(range(num_phases))

    return {
        "total_misclassified": len(misclassified_indices),
        "per_phase": {phase: int(counts.get(phase, 0)) for phase in phases}
    }


def save_per_phase_metrics(acc, precision, recall, f1, cm, misclassified_info, out_file):
    """
    Save eval metrics to specified JSON file.
    """

    with open(out_file, "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
                "misclassified_samples": misclassified_info,
            },
            f,
            indent=2,
        )
    print("Saved metrics to:", out_file)


def save_metrics(acc, precision, recall, f1, cm, out_file):
    """
    Save eval metrics to specified JSON file.
    """

    with open(out_file, "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )
    print("Saved metrics to:", out_file)