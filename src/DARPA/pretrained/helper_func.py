"""
Helper functions for loading datasets, evaluating models, and analyzing misclassifications.
"""

import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def load_datasets(dataset_dir, phase, sparse=False):
    """
    Load DARPA datasets for a specific attack phase.
    
    :param dataset_dir: Directory containing the dataset files
    :param phase: Attack phase (1-5)
    :param sparse: Whether the feature matrices are stored in sparse format
    """
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


def evaluate(model, dataloader):
    """
    Evaluate a PyTorch model on a given dataloader.
    
    :param model: PyTorch model to evaluate
    :param dataloader: DataLoader providing the evaluation data
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
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


def save_metrics(phase, acc, precision, recall, f1, cm, misclassified_info, output_dir):
    """
    Save evaluation metrics to a JSON file in the specified output directory.
    """
    metrics_file = output_dir / f"phase_{phase}.json"
    with open(metrics_file, "w") as f:
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
    print("Saved metrics to:", metrics_file)


def save_loss_plot(train_losses, phase, epochs, output_dir):
    """
    Save the training loss plot to the specified output directory.
    """
    print("\nSaving training loss plot...")
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    figure_path = output_dir / f"phase_{phase}_training_loss.png" 
    plt.savefig(figure_path, dpi=300)  
    plt.close()  

    print(f"Training loss plot saved to: {figure_path}\n")