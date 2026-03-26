import numpy as np
import matplotlib.pyplot as plt


def plot_ids_confusion_matrix(
    cm,
    classes,
    normalize=True,
    title="Confusion Matrix",
    savepath=None,
):
    """
    IDS-style confusion matrix visualization.

    Assumes:
        cm[predicted, actual]
    """

    cm = np.asarray(cm, dtype=float)

    # -------------------------
    # Normalize per actual class
    # -------------------------
    if normalize:
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm_display = cm / col_sums
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm_display, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Predicted",
        xlabel="Actual",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # -------------------------
    # Annotate cells
    # -------------------------
    thresh = cm_display.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_display[i, j]
            text = f"{val:.3f}" if normalize else int(cm[i, j])

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    # -------------------------
    # Highlight IDS errors
    # -------------------------
    benign_idx = classes.index("benign")

    for i in range(len(classes)):
        for j in range(len(classes)):

            # False alarm: benign → attack
            if j == benign_idx and i != benign_idx:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )

            # Missed attack: attack → benign
            if i == benign_idx and j != benign_idx:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="orange",
                        linewidth=2,
                    )
                )

    fig.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def compute_ids_rates(cm, classes):
    """
    Compute Detection Rate and False Alarm Rate
    from confusion matrix (predicted, actual).
    """

    cm = np.asarray(cm, dtype=float)

    benign_idx = classes.index("benign")

    # ---------------------
    # False Alarm Rate
    # ---------------------
    total_benign = cm[:, benign_idx].sum()
    benign_correct = cm[benign_idx, benign_idx]

    false_alarms = total_benign - benign_correct
    far = false_alarms / total_benign if total_benign else 0.0

    # ---------------------
    # Detection Rate
    # ---------------------
    attack_idxs = [i for i in range(len(classes)) if i != benign_idx]

    TP = sum(cm[i, i] for i in attack_idxs)
    FN = sum(
        cm[benign_idx, j]
        for j in attack_idxs
    )

    detection_rate = TP / (TP + FN) if (TP + FN) else 0.0

    return detection_rate, far


def plot_detection_vs_far(results, title="Detection vs False Alarm Rate"):
    """
    results = [
        ("Model name", detection_rate, far),
        ...
    ]
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, dr, far in results:
        ax.scatter(far, dr, s=120)
        ax.text(far, dr, " " + name, fontsize=10, va="center")

    ax.set_xlabel("False Alarm Rate (FAR)")
    ax.set_ylabel("Detection Rate")
    ax.set_title(title)

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)

    ax.grid(True, linestyle="--", alpha=0.5)

    # Ideal region indicator
    ax.annotate(
        "Ideal",
        xy=(0.02, 0.98),
        xytext=(0.15, 0.85),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.tight_layout()
    plt.show()