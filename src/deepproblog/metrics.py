from pathlib import Path

import numpy as np

from src.datasets.flow_datasets import SubsetDPLDataset


def all_phases_indices(dataset):
    """ 
    Identify indices in the original dataset where all prev-phase flags are 1. 
    """
    filtered_idx = []
    for i in range(len(dataset)):
        # data entries are: [curr_phase, p1, p2, p3, p4, label]
        entry = dataset.data[i]
        _, p1, p2, p3, p4, _ = entry
        if int(p1) == 1 and int(p2) == 1 and int(p3) == 1 and int(p4) == 1:
            filtered_idx.append(i)
    return filtered_idx


def get_filtered_dataset(dataset, filter_name):
    """ 
    Return a subset of the dataset based on the filter_name. 
    Currently supports 'all_prev_phases' filter.
    """
    if filter_name == "all_prev_phases":
        filtered_idx = all_phases_indices(dataset)

        if len(filtered_idx) == 0:
            print("No test examples have all previous phases present; returning None.")
            return None

        filtered_set = SubsetDPLDataset(dataset, filtered_idx)
        return filtered_set
    
    else:
        print(f"Unknown filter name: {filter_name}; returning None.")
        return None


def snapshot_params(net):
    """ Take a snapshot of the parameters of a PyTorch module. """
    return {n: p.detach().cpu().clone() for n, p in net.named_parameters()}


def print_param_changes(modules, snapshots_before):
    """ Print parameter changes for monitored PyTorch modules. """
    for i, net in enumerate(modules):
        before = snapshots_before[i]
        after = snapshot_params(net)  
        for name in before:
            diff = (after[name] - before[name]).norm().item()
            before_norm = before[name].norm().item() or 1e-12
            rel = diff / before_norm
            print(f"{name}: L2-change={diff:.6e}, relative={rel:.4%}")


def compute_metrics_from_cm(cm):
    """
    Compute metrics from a DeepProbLog ConfusionMatrix.
    Supports binary and multi-class cases.

    Confusion matrix format:
        matrix[predicted_index, actual_index]
    """

    mat = getattr(cm, "matrix", None)
    classes = list(getattr(cm, "classes", []))

    if mat is None or len(classes) == 0:
        print("Confusion matrix is empty.")
        return None

    mat = np.asarray(mat, dtype=float)
    n_classes = len(classes)
    total = mat.sum()

    results = {
        "classes": classes,
        "total": int(total),
    }

    # -------- Overall accuracy --------
    correct = np.trace(mat)
    accuracy = correct / total if total else 0.0
    results["accuracy"] = accuracy

    # -------- Per-class metrics --------
    per_class = {}
    supports = mat.sum(axis=0)  # actual counts per class

    precisions = []
    recalls = []
    f1s = []
    weights = []

    for i, cls in enumerate(classes):
        TP = mat[i, i]
        FP = mat[i, :].sum() - TP
        FN = mat[:, i].sum() - TP
        TN = total - TP - FP - FN

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        support = supports[i]

        per_class[cls] = {
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "TN": int(TN),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weights.append(support)

    results["per_class"] = per_class

    # -------- Macro metrics --------
    results["macro_precision"] = float(np.mean(precisions))
    results["macro_recall"] = float(np.mean(recalls))
    results["macro_f1"] = float(np.mean(f1s))

    # -------- Weighted metrics --------
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        results["weighted_f1"] = float(
            np.sum(np.array(f1s) * np.array(weights)) / weight_sum
        )
    else:
        results["weighted_f1"] = 0.0

    # -------- Micro metrics --------
    TP_micro = correct
    FP_micro = mat.sum(axis=1).sum() - correct
    FN_micro = mat.sum(axis=0).sum() - correct

    micro_precision = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) else 0.0
    micro_recall = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    results["micro_precision"] = micro_precision
    results["micro_recall"] = micro_recall
    results["micro_f1"] = micro_f1

    # -------- Binary-specific convenience fields --------
    if n_classes == 2:
        pos_label = classes[-1]
        results["positive_class"] = pos_label
        
        # Copy class-specific metrics
        results.update(per_class[pos_label])

        # Compute specificity = TN / (TN + FP)
        TN = per_class[pos_label]["TN"]
        FP = per_class[pos_label]["FP"]
        results["specificity"] = TN / (TN + FP) if (TN + FP) else 0.0

    return results


def log_metrics(logger, metrics, title=None, per_class=True):
    if title:
        logger.comment(f"\n=== {title} ===")

    # ---- Global metrics ----
    logger.comment(f"\nAccuracy: {metrics['accuracy']:.4f}")

    if "micro_f1" in metrics:
        logger.comment(
            f"Micro F1: {metrics['micro_f1']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Weighted F1: {metrics['weighted_f1']:.4f}"
        )

    # ---- Binary-only metrics ----
    if "positive_class" in metrics:
        logger.comment(
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Specificity: {metrics['specificity']:.4f}"
        )

    # ---- Per-class metrics ----
    if per_class and "per_class" in metrics:
        logger.comment("\nPer-class metrics:")
        for cls, m in metrics["per_class"].items():
            logger.comment(
                f"  [{cls}] "
                f"P={m['precision']:.4f} | "
                f"R={m['recall']:.4f} | "
                f"F1={m['f1']:.4f} | "
                f"Support={m['support']}"
            )