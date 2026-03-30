import json
from collections import Counter

import numpy as np
import torch

from sklearn.metrics import confusion_matrix


def eval(
    model,
    dataloader,
    multi_class: bool = True,
    device: torch.device | None = None,
):
    """
    Evaluate a PyTorch model on a given dataloader.

    :param model: The PyTorch model to evaluate.
    :param dataloader: DataLoader providing the evaluation data.
    :param multi_class: Whether the model is multi-class (True) or binary (False).
    :param device: The device to run evaluation on (e.g., "cpu" or "cuda"). If None, uses the model's device.
    
    :return: A tuple containing the confusion matrix and the predicted labels.
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

    labels = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    classes = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    if multi_class:
        classes = ["benign" if c == 0 else f"phase{c}" for c in classes]
    else:
        classes = ["benign", "attack"]

    return cm, classes, y_pred


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


def save_per_phase_metrics(cm, metrics, mis_info, out_file):
    """
    Save per phase eval metrics to JSON file.
    """
    with open(out_file, "w") as f:
        json.dump(
            {
                "Accuracy": metrics["accuracy"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "False Alarms": metrics["false_alarms"],
                "False Alarm Rate": metrics["false_alarm_rate"],
                "Missed Attacks": metrics["missed_attacks"],
                "Detection Rate": metrics["detection_rate"],
                "Confusion Matrix (actual_pred)": cm.tolist(),

                "Misclassified Info": mis_info,
            },
            f,
            indent=2,
        )
    print("Saved metrics to:", out_file)


def save_metrics(cm, metrics, mis_indices, real_indices, y_pred, y_true, out_file):
    """
    Save eval metrics to JSON file.
    """

    with open(out_file, "w") as f:
        json.dump(
            {
                "Accuracy": metrics["accuracy"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "False Alarms": metrics["false_alarms"],
                "False Alarm Rate": metrics["false_alarm_rate"],
                "Missed Attacks": metrics["missed_attacks"],
                "Detection Rate": metrics["detection_rate"],
                "Confusion Matrix (actual_pred)": cm.tolist(),

                "missclassified_indices": mis_indices.tolist() if isinstance(mis_indices, np.ndarray) else mis_indices,
                "real_flow_indices": real_indices.tolist() if isinstance(real_indices, np.ndarray) else real_indices,
                "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
                "y_true": y_true.tolist() if isinstance(y_true, np.ndarray) else y_true,
            },
            f,
            indent=2,
        )
    print("Saved metrics to:", out_file)