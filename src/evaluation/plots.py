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

    diag_mask = np.eye(n, dtype=bool)

    fp_mask = np.zeros_like(cm, dtype=bool)
    fn_mask = np.zeros_like(cm, dtype=bool)

    for i in range(n):
        for j in range(n):
            if j == benign_idx and i != benign_idx:
                fp_mask[i, j] = True   # false alarm
            elif i == benign_idx and j != benign_idx:
                fn_mask[i, j] = True   # missed attack
    
    return diag_mask, fp_mask, fn_mask


def plot_confusion_matrix(
    cm,
    classes,
    experiment_name,
    out_path,
):
    """
    IDS-style confusion matrix visualization.

    Assumes:
        cm[predicted, actual]
    """

    cm_display = np.asarray(cm, dtype=float) + 1

    diag_mask, fp_mask, fn_mask = compute_masks(cm_display, classes)
    masks = {"TP": diag_mask, "FP": fp_mask, "FN": fn_mask}
    cm_colors = {"TP": "Greens", "FP": "Reds", "FN": "Purples"}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm_display, cmap="Greys", norm=LogNorm(), interpolation="none")

    for label, mask in masks.items():
        ax.imshow(
            np.ma.masked_where(~mask, cm_display),
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
        title=f"Confusion Matrix - {experiment_name}",
    )

    ax.set_xticks(np.arange(len(classes)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(classes)+1)-.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # -------------------------
    # Annotate cells
    # -------------------------
    thresh = 20 # cm_display.max() / 10 # np.median(cm_display)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            val = cm_display[i, j]
            count = int(cm[i, j])

            # Decide label
            label = ""
            if diag_mask[i, j]:
                label = "TP"
            elif fp_mask[i, j]:
                label = "FP"
            elif fn_mask[i, j]:
                label = "FN"

            text_color = "white" if val > thresh else "black"

            ax.text(
                j,
                i,
                f"{count}\n{label}" if label else f"{count}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                # fontweight="bold" if label else "normal",
            )

    fig.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved confusion matrix plot to:", out_path)