"""
Helper functions for evaluating models, and analyzing misclassifications.
"""

import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate(model, dataloader, average: str = "auto"):
    """
    Evaluate a PyTorch model on a given dataloader.

    This helper supports both binary and multi-class classification. By default
    `average="auto"` chooses 'binary' when two classes are present and
    'macro' for multi-class problems. You can override it by passing a
    specific averaging method accepted by
    `sklearn.metrics.precision_recall_fscore_support` (e.g. 'macro', 'micro',
    'weighted', or 'binary').

    :param model: PyTorch model to evaluate
    :param dataloader: DataLoader providing the evaluation data
    :param average: Averaging method for precision/recall/F1 or 'auto'
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            probs = model(X_batch)      # [batch, num_classes]

            # Convert probabilities/logits to predicted labels
            preds = probs.argmax(dim=1) # [batch]

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)

    # Decide averaging strategy: use binary for 2 classes, otherwise macro (unless overridden)
    unique_labels = np.unique(np.concatenate([y_true, y_pred])) if y_true.size or y_pred.size else np.array([])
    if average == "auto":
        avg = "binary" if unique_labels.size == 2 else "macro"
    else:
        avg = average

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

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


def save_metrics(acc, precision, recall, f1, cm, misclassified_info, out_file):
    """
    Save eval metrics to specified JSON file.
    """
    if misclassified_info is None:
        misclassified_info = {}

    with open(out_file, "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
                "misclassified_info": misclassified_info
            },
            f,
            indent=2,
        )
    print("Saved metrics to:", out_file)


def save_confusion_matrix_heatmap(
    cm,
    class_names,
    title,
    out_file,
):
    """
    Save a confusion matrix heatmap (sklearn-style).

    Parameters
    ----------
    cm : array-like (n_classes, n_classes)
        Confusion matrix from sklearn.metrics.confusion_matrix
        Layout: rows = actual, columns = predicted
    class_names : list[str]
        Class labels in the same order used to compute cm
    title : str
        Plot title
    out_file : str
        Save the plot to this file instead of displaying it.
    """

    cm = np.asarray(cm, dtype=float)

    plt.figure()
    plt.imshow(cm)
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def save_loss_plot(train_losses, epochs, out_file):
    """
    Save training loss plot to specified output file.

    :param train_losses: List of training losses per epoch
    :param epochs: Total number of training epochs
    :param out_file: Output file path to save the plot
    """
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.savefig(out_file, dpi=300)  
    plt.close()  

    print(f"Training loss plot saved to: {out_file}")