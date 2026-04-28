from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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


def plot_train_loss(logger, out_path):

    # ---- Extract loss from logger ----
    if not logger.has_attribute("loss"):
        print("No loss logged.")
        return

    indices, losses = logger["loss"]  # GETTER shorthand

    # ---- Build DataFrame ----
    df = pd.DataFrame({"i": indices, "loss": losses})
    df = df.sort_values("i")

    # ---- Smooth loss (moving average) ----
    window = max(5, len(df) // 20)  # adaptive window
    df["loss_smooth"] = df["loss"].rolling(window, min_periods=1).mean()

    # ---- Plot ----
    plt.figure(figsize=(8, 5))
    plt.plot(df["i"], df["loss"], alpha=0.3, label="Loss")
    plt.plot(df["i"], df["loss_smooth"], linewidth=2, label="Smoothed Loss")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ---- Save ----
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved loss plot to:", out_path)


def compute_masks(cm, classes):
    n = len(classes)
    benign_idx = classes.index("benign")

    diag_mask = np.zeros_like(cm, dtype=bool)
    fp_mask = np.zeros_like(cm, dtype=bool)
    fn_mask = np.zeros_like(cm, dtype=bool)
    off_diag_mask = np.zeros_like(cm, dtype=bool)

    for i in range(n):
        for j in range(n):
            # correct prediction
            if i == j:
                diag_mask[i, j] = True
            # false alarm
            elif j == benign_idx and i != benign_idx:
                fp_mask[i, j] = True   
            # missed attack
            elif i == benign_idx and j != benign_idx:
                fn_mask[i, j] = True   
            # off diagonal but not involving benign
            else:
                off_diag_mask[i, j] = True
            
    return diag_mask, fp_mask, fn_mask, off_diag_mask


def plot_confusion_matrix(
    cm,
    classes,
    out_path,
):
    """
    IDS-style confusion matrix visualization.

    Assumes:
        cm[predicted, actual]
    """

    cm = np.asarray(cm, dtype=float)

    # Normalize by column to get fractions
    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1  # avoid division by zero
    cm_normalized = cm / col_sums  # values are 0–1 fractions

    diag_mask, fp_mask, fn_mask, off_diag_mask = compute_masks(cm, classes)
    masks = {"TP": diag_mask, "FP": fp_mask, "FN": fn_mask, "OFF-DIAG": off_diag_mask}
    cm_colors = {"TP": "Greens", "FP": "Oranges", "FN": "Oranges", "OFF-DIAG": "Reds"}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm_normalized, cmap="Greys", norm=LogNorm(), interpolation="none")

    for label, mask in masks.items():
        ax.imshow(
            np.ma.masked_where(~mask, cm_normalized),
            cmap=cm_colors[label],
            norm=LogNorm(),
            interpolation="none",
            alpha=0.85,
        )
    
    ax.set_aspect("equal")

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Predicted",
        xlabel="Actual",
    )

    ax.set_xticks(np.arange(len(classes)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(classes)+1)-.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            frac = cm_normalized[i, j]
            count = int(cm[i, j])

            text_color = "white" if frac > thresh else "black"

            ax.text(
                j,
                i,
                f"{count}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved confusion matrix plot to:", out_path)