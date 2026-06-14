from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_train_loss(train_losses, epochs, out_file):
    """
    Save training loss plot to specified output file.
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


def plot_train_val_loss(train_losses, val_losses, out_file, title="Training and Validation Loss"):
    """Plot training and validation loss per epoch and save to out_file.

    train_losses: list of floats
    val_losses: list of floats or None (same length as train_losses or empty)
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss', color='C0')

    if val_losses is not None and len(val_losses) == len(train_losses):
        plt.plot(epochs, val_losses, marker='o', label='Val Loss', color='C1')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved train/val loss plot to: {out_file}")


def plot_train_loss_dpl(logger, out_path):
    """
    Plot loss curve from DeepProbLog training logs.
    """

    # ---- Extract loss from logger ----
    if not logger.has_attribute("loss"):
        print("No loss logged.")
        return

    indices, losses = logger["loss"] 

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


def plot_confusion_matrix_2x5(
    y_pred,
    y_test,
    classes,
    true_classes,
    pred_classes,
    out_path,
):
    """
    IDS-style confusion matrix for binary predictions vs multi-class ground truth.

    Assumes:
        cm[predicted, actual] with shape (2, 5)
        predicted: 0=Normal, 1=Anomaly
        actual: 0=Benign, 1-4=Attack classes
    """

    # Build the 2x5 matrix manually
    cm_2x5 = np.zeros((2, len(classes)), dtype=int)
    for pred_label in [0, 1]:
        for true_class in classes:
            cm_2x5[pred_label, true_class] = np.sum((y_pred == pred_label) & (y_test == true_class))
        
    cm = np.asarray(cm_2x5, dtype=float)

    # Normalize by column
    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_normalized = cm / col_sums

    # Define masks for 2x5 matrix
    # TP: predicted anomaly (row 1) AND actual anomaly (cols 1-4)
    # TN: predicted normal (row 0) AND actual benign (col 0)
    # FP: predicted anomaly (row 1) AND actual benign (col 0)
    # FN: predicted normal (row 0) AND actual anomaly (cols 1-4)
    
    tp_mask = np.zeros_like(cm, dtype=bool)
    tp_mask[1, 1:] = True  # row 1, cols 1-4

    tn_mask = np.zeros_like(cm, dtype=bool)
    tn_mask[0, 0] = True  # row 0, col 0

    fp_mask = np.zeros_like(cm, dtype=bool)
    fp_mask[1, 0] = True  # row 1, col 0

    fn_mask = np.zeros_like(cm, dtype=bool)
    fn_mask[0, 1:] = True  # row 0, cols 1-4

    masks = {"TP": tp_mask, "TN": tn_mask, "FP": fp_mask, "FN": fn_mask}
    cm_colors = {"TP": "Greens", "TN": "Greens", "FP": "Oranges", "FN": "Oranges"}

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Handle zeros for LogNorm
    cm_plot = np.where(cm_normalized == 0, np.nan, cm_normalized)
    ax.imshow(cm_plot, cmap="Greys", norm=LogNorm(), interpolation="none")

    for label, mask in masks.items():
        masked_data = np.ma.masked_where(~mask, cm_plot)
        ax.imshow(
            masked_data,
            cmap=cm_colors[label],
            norm=LogNorm(),
            interpolation="none",
            alpha=0.85,
        )

    ax.set(
        xticks=np.arange(len(true_classes)),
        yticks=np.arange(len(pred_classes)),
        xticklabels=true_classes,
        yticklabels=pred_classes,
        xlabel="Actual",
        ylabel="Predicted",
    )

    ax.set_xticks(np.arange(len(true_classes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pred_classes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = np.nanmax(cm_normalized) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            frac = cm_normalized[i, j]
            count = int(cm[i, j])
            text_color = "white" if frac > thresh else "black"
            ax.text(j, i, f"{count}", ha="center", va="center", color=text_color, fontsize=10)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved confusion matrix plot to:", out_path)


def make_dir(experiment_dir, subpath):
    path = Path(experiment_dir) / subpath
    path.mkdir(parents=True, exist_ok=True)

    return path


def save_plots(experiment_dir, experiment_name, run_id, logger, cm, classes):

    loss_plot_dir = make_dir(experiment_dir, "loss_plots")
    plot_train_loss_dpl(
        logger=logger,
        out_path = loss_plot_dir / f"{experiment_name}_{run_id}.png",
    )

    cm_dir = make_dir(experiment_dir, "cm_plots")
    plot_confusion_matrix(
        cm=cm,
        classes=classes,
        out_path = cm_dir / f"{experiment_name}_{run_id}.png",
    )