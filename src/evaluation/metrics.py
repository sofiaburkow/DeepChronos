import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_metrics(y_true, y_pred, confusion_matrix, classes, layout="actual_pred"):
    """
    Compute evaluation metrics from confusion matrix and predictions.
    """
    cm = np.asarray(confusion_matrix, dtype=float)
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    if layout not in {"actual_pred", "pred_actual"}:
        raise ValueError("layout must be 'actual_pred' or 'pred_actual'")
    if layout == "actual_pred":
        cm = cm.T # transpose to get [predicted, actual] orientation

    classes = list(classes)
    if "benign" not in classes:
        raise ValueError("Classes must contain 'benign' label.")

    total = cm.sum()
    accuracy = np.trace(cm) / total if total else 0.0
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    per_class = {}
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = cm[i, :].sum() - TP
        FN = cm[:, i].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_class[cls] = dict(
            TP=int(TP),
            FP=int(FP),
            FN=int(FN),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
        )

    # False Alarm Rate and Detection Rate
    benign_idx = classes.index("benign")
    n_classes = len(classes)
    attack_idxs = [i for i in range(n_classes) if i != benign_idx]

    total_benign = cm[:, benign_idx].sum()
    benign_correct = cm[benign_idx, benign_idx]
    false_alarms = total_benign - benign_correct
    false_alarm_rate = false_alarms / total_benign if total_benign else 0.0

    total_attacks = cm[:, attack_idxs].sum()
    detected_attacks = sum(
        cm[:, i].sum() - cm[benign_idx, i]
        for i in attack_idxs
    )
    missed_attacks = total_attacks - detected_attacks
    detection_rate = (
        detected_attacks / total_attacks
        if total_attacks > 0
        else 0.0
    )

    return dict(
        accuracy=float(accuracy),
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        false_alarms=int(false_alarms),
        false_alarm_rate=float(false_alarm_rate),
        missed_attacks=int(missed_attacks),
        detection_rate=float(detection_rate),
        classes=classes,
        per_class=per_class,
    )


def aggregate_fold_metrics(fold_metrics):
    """Aggregate per-fold metric dicts into one summary dict (same structure as compute_metrics output)."""
    if not fold_metrics:
        return {}

    summary = {}
    summary["accuracy"] = np.mean([m["accuracy"] for m in fold_metrics])
    summary["micro_f1"] = np.mean([m["micro_f1"] for m in fold_metrics])
    summary["macro_f1"] = np.mean([m["macro_f1"] for m in fold_metrics])
    summary["false_alarm_rate"] = np.mean([m["false_alarm_rate"] for m in fold_metrics])
    summary["detection_rate"] = np.mean([m["detection_rate"] for m in fold_metrics])

    return summary


def save_per_phase_metrics_json(cm, metrics, mis_info, out_file):
    """
    Save per-phase evaluation metrics and confusion matrix to JSON.
    """
    data = dict(metrics)
    data["confusion_matrix"] = cm.tolist()
    data["misclassified_indices"] = mis_info
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved metrics to: {out_file}")


def save_metrics_json(cm, metrics, out_file):
    """
    Save evaluation metrics and confusion matrix to JSON.
    """
    data = dict(metrics)
    data["confusion_matrix"] = np.asarray(cm).tolist()
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved metrics to: {out_file}")


def extract_cm(cm):
    """
    Convert DeepProbLog or tensor CM to numpy array + classes.
    """
    classes = None
    if hasattr(cm, "matrix"):
        classes = list(getattr(cm, "classes", []))
        mat = cm.matrix
    else:
        mat = cm

    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()
    mat = np.asarray(mat, dtype=float)

    return mat, classes


def make_dir(experiment_dir, logic_file, subpath):
    path = Path(experiment_dir) / logic_file / subpath
    path.mkdir(parents=True, exist_ok=True)
    
    return path


def log_metrics(logger, experiment_dir, logic_file, experiment_name, run_id, metrics, per_class, inference_times, cm):

    logs_dir = make_dir(experiment_dir, logic_file, "logs")
    log_file = logs_dir / f"{experiment_name}_{run_id}"

    logger.comment(f"=== {experiment_name} ===")
    logger.comment("\nMetrics:")
    logger.comment(
        f"  Accuracy: {metrics['accuracy']:.4f} | "
        f"Micro F1: {metrics['micro_f1']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )
    logger.comment(
        f"  False Alarms: {metrics['false_alarms']} | "
        f"False Alarm Rate: {metrics['false_alarm_rate']:.4f}"
    )
    logger.comment(
        f"  Missed Attacks: {metrics['missed_attacks']} | "
        f"Detection Rate: {metrics['detection_rate']:.4f}"
    )

    if per_class:
        logger.comment("\nPer-Class Metrics:")

        for cls, m in metrics["per_class"].items():
            if cls == "benign": 
                continue  

            logger.comment(
                f"  [{cls}] "
                f"P={m['precision']:.4f} | "
                f"R={m['recall']:.4f} | "
                f"F1={m['f1']:.4f} | "
            )
    
    logger.comment(
        f"\nAverage inference time: {np.mean(inference_times):.4f}"
        f"Total inference time: {np.sum(inference_times):.4f}"
    )
    logger.comment("\nConfusion Matrix:\n" + str(cm))

    logger.write_to_file(str(log_file))
    print("Saved log to:", f"{log_file}.log")


def save_dpl_metrics(
    experiment_dir,
    logic_file,
    experiment_name,
    run_id,
    cm,
    metrics,
    classes,
    inference_times,
    errors,
    correct,
):
    """Save all evaluation artifacts (metrics, plots, model, logs)."""

    # --- Metrics ---
    metrics_dir = make_dir(experiment_dir, logic_file, "metrics")
    np.savez(
        metrics_dir / f"{experiment_name}_{run_id}.npz",
        confusion_matrix=cm,
        classes=np.array(classes),
        metrics=metrics,
        inference_times=np.array(inference_times),
    )

    # --- Errors / correct classifications ---
    for name, data in [("errors", errors), ("correct", correct)]:
        out_dir = make_dir(experiment_dir, logic_file, name)
        out_path = out_dir / f"{experiment_name}_{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    print(f"\Metrics saved under {Path(experiment_dir) / logic_file}")
