from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.dpl_metrics import extract_cm


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


def plot_train_loss(logger, plot_dir, experiment_name, run_id):

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
    plt.title("DeepProbLog Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ---- Save ----
    plot_path = plot_dir / f"{experiment_name}_{run_id}_loss.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved loss plot:", plot_path)


def plot_confusion_matrix(
    cm,
    plot_dir,
    experiment_name,
    run_id,
    log_scale=True,
):
    """
    Confusion matrix visualization.

    Assumes:
        cm[predicted, actual]
    """

    cm, classes = extract_cm(cm)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Color scaling
    display_cm = cm.copy()

    # if log_scale:
    #     display_cm = np.log1p(display_cm)  # handles large imbalance

    im = ax.imshow(display_cm, cmap="Blues")
    
    # Labels
    # -----------------------------
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{experiment_name} Confusion Matrix", fontsize=14, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate counts
    max_val = cm.max()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            value = int(cm[i, j])
            if value == 0:
                continue

            color = "white" if value > max_val * 0.5 else "black"

            ax.text(
                j, i,
                f"{value:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold" if i == j else "normal",
            )

    # Highlight IDS errors
    benign_idx = classes.index("benign")

    for i in range(len(classes)):
        for j in range(len(classes)):

            # False alarms
            if j == benign_idx and i != benign_idx:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
                ax.add_patch(rect)

            # Missed attacks
            if i == benign_idx and j != benign_idx:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    fill=False,
                    edgecolor="orange",
                    linewidth=2,
                )
                ax.add_patch(rect)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # ---- Save ----
    plot_path = plot_dir / f"{experiment_name}_{run_id}_cm.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved confusion matrix plot:", plot_path)