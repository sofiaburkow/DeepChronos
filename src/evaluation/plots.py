from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def plot_dpl_train_loss(logger, experiment_dir, experiment_name, run_id):

    # --- Convert logs ---
    df = pd.DataFrame(logger.log)

    if "loss" not in df.columns:
        print("No loss logged.")
        return

    df = df.sort_values("i")

    # --- Smooth loss (moving average) ---
    df["loss_smooth"] = df["loss"].rolling(window=50, min_periods=1).mean()

    # --- Plot ---
    plt.figure(figsize=(8, 5))

    # raw loss (light)
    plt.plot(df["i"], df["loss"], alpha=0.3, label="Loss")

    # smoothed loss
    plt.plot(df["i"], df["loss_smooth"], linewidth=2, label="Smoothed Loss")

    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("DeepProbLog Training Loss")

    plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # --- Save ---
    plot_dir = Path(experiment_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / f"{experiment_name}_{run_id}_loss.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved loss plot:", plot_path)