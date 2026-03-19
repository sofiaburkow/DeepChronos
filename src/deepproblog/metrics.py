import numpy as np
import torch

from src.datasets.flow_datasets import SubsetDPLDataset

from typing import Optional

from deepproblog.dataset import Dataset
from deepproblog.model import Model
from deepproblog.utils.confusion_matrix import ConfusionMatrix



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
    Compute multi-class metrics from a DeepProbLog ConfusionMatrix
    or raw matrix.

    Expected layout:
        matrix[predicted, actual]
    """

    # Normalize input
    classes = None

    # DeepProbLog ConfusionMatrix object
    if hasattr(cm, "matrix"):
        classes = list(getattr(cm, "classes", []))
        mat = cm.matrix

    else:
        mat = cm

    # torch → numpy
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()

    mat = np.asarray(mat, dtype=float)

    if mat.ndim != 2:
        raise ValueError(f"Confusion matrix must be 2D, got {mat.shape}")

    n_classes = mat.shape[0]
    total = mat.sum()

    # fallback class names
    if not classes:
        classes = [str(i) for i in range(n_classes)]

    # Overall accuracy
    accuracy = np.trace(mat) / total if total else 0.0

    # Overall FPR (micro)
    FP_total = 0
    TN_total = 0

    # Per-class metrics
    per_class = {}

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, cls in enumerate(classes):

        TP = mat[i, i]
        FP = mat[i, :].sum() - TP
        FN = mat[:, i].sum() - TP
        TN = total - TP - FP - FN

        FP_total += FP
        TN_total += TN

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        fnr = FN / (TP + FN) if (TP + FN) else 0.0
        fpr = FP / (FP + TN) if (FP + TN) else 0.0

        support = mat[:, i].sum()

        per_class[cls] = {
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "TN": int(TN),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "support": int(support),
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    # Aggregates
    supports = np.array(supports)

    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports))

    # micro metrics
    correct = np.trace(mat)
    micro_precision = correct / total if total else 0.0
    micro_recall = micro_precision
    micro_f1 = micro_precision
    overall_fpr = FP_total / (FP_total + TN_total) if (FP_total + TN_total) else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "overall_fpr": overall_fpr,
        "classes": classes,
        "per_class": per_class,
    }


def log_metrics(logger, metrics, title=None, per_class=True):
    if title:
        logger.comment(f"\n=== {title} ===")

    # Global metrics
    logger.comment(f"\nAccuracy: {metrics['accuracy']:.4f}")

    if "micro_f1" in metrics:
        logger.comment(
            f"Micro F1: {metrics['micro_f1']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Weighted F1: {metrics['weighted_f1']:.4f} | "
            f"Overall FPR: {metrics['overall_fpr']:.4f}"
        )

    # Per-class metrics
    if per_class and "per_class" in metrics:
        logger.comment("\nPer-class metrics:")

        for cls, m in metrics["per_class"].items():

            precision = m.get("precision", 0.0)
            recall = m.get("recall", 0.0)
            f1 = m.get("f1", 0.0)
            fpr = m.get("fpr", None)
            fnr = m.get("fnr", None)
            support = m.get("support", 0)

            line = (
                f"  [{cls}] "
                f"P={precision:.4f} | "
                f"R={recall:.4f} | "
                f"F1={f1:.4f}"
            )

            # Add new metrics if present
            if fpr is not None:
                line += f" | FPR={fpr:.6f}"
            if fnr is not None:
                line += f" | FNR={fnr:.6f}"

            line += f" | Support={support}"

            logger.comment(line)


def get_confusion_matrix(
    model: Model, dataset: Dataset, verbose: int = 0, eps: Optional[float] = None
):
    confusion_matrix = ConfusionMatrix()
    misclassified = []

    model.eval()

    for i, gt_query in enumerate(dataset.to_queries()):
        test_query = gt_query.variable_output()
        answer = model.solve([test_query])[0]
        actual = str(gt_query.output_values()[0])

        if len(answer.result) == 0:
            predicted = "no_answer"
            p = None
        else:
            max_ans = max(answer.result, key=lambda x: answer.result[x])
            p = answer.result[max_ans]

            if eps is None:
                predicted = str(max_ans.args[gt_query.output_ind[0]])
            else:
                predicted = float(max_ans.args[gt_query.output_ind[0]])
                actual = float(gt_query.output_values()[0])
                if abs(actual - predicted) < eps:
                    predicted = actual

        if actual != predicted:
            misclassified.append({
                "index": i,
                "actual": actual,
                "predicted": predicted,
                "confidence": p,
                "test_query": test_query,
            })

        confusion_matrix.add_item(predicted, actual)

    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix, misclassified