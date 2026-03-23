import numpy as np
import torch

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
    Compute multi-class metrics & security metrics 
    from a DeepProbLog ConfusionMatrix.

    Expected layout:
        matrix[predicted, actual]
    """

    # ---------------------------
    # Normalize input
    # ---------------------------
    classes = None

    if hasattr(cm, "matrix"):
        classes = list(getattr(cm, "classes", []))
        mat = cm.matrix
    else:
        mat = cm

    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()

    mat = np.asarray(mat, dtype=float)

    if mat.ndim != 2:
        raise ValueError(f"Confusion matrix must be 2D, got {mat.shape}")

    n_classes = mat.shape[0]
    total = mat.sum()

    if not classes:
        classes = [str(i) for i in range(n_classes)]

    # ---------------------------
    # Global accuracy
    # ---------------------------
    accuracy = np.trace(mat) / total if total else 0.0

    # ---------------------------
    # Per-class metrics
    # ---------------------------
    per_class = {}
    f1s = []
    supports = []

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

        f1s.append(f1)
        supports.append(support)

    # Aggregates
    supports = np.array(supports)

    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports))

    # ---------------------------
    # IDS / Security Metrics
    # ---------------------------
    benign_label = "benign"
    benign_idx = classes.index(benign_label)
    attack_idxs = [i for i in range(len(classes)) if i != benign_idx]

    total_benign = mat[:, benign_idx].sum()
    benign_correct = mat[benign_idx, benign_idx]

    benign_as_attack = total_benign - benign_correct

    false_alarm_rate = (
        benign_as_attack / total_benign if total_benign else 0.0
    )

    total_attacks = mat[:, attack_idxs].sum()

    attack_correct = sum(mat[i, i] for i in attack_idxs)

    attack_detection_rate = (
        attack_correct / total_attacks if total_attacks else 0.0
    )

    missed_attacks = mat[benign_idx, attack_idxs].sum()

    missed_attack_rate = (
        missed_attacks / total_attacks if total_attacks else 0.0
    )

    security = dict(
        false_alarm_rate=false_alarm_rate,
        benign_as_attack=int(benign_as_attack),
        attack_detection_rate=attack_detection_rate,
        missed_attack_rate=missed_attack_rate,
    )

    # ---------------------------
    return dict(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        classes=classes,
        per_class=per_class,
        security=security,
    )


def log_metrics(logger, metrics, title=None, per_class=True):

    if title:
        logger.comment(f"\n=== {title} ===")

    logger.comment(
        f"\nAccuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )

    # ---------------------------
    # Security metrics
    # ---------------------------
    sec = metrics.get("security", {})
    if sec:
        logger.comment(
            "\nSecurity Metrics:"
            f"\n  False Alarm Rate: {sec['false_alarm_rate']:.6f}"
            f"\n  Attack Detection Rate: {sec['attack_detection_rate']:.6f}"
            f"\n  Missed Attack Rate: {sec['missed_attack_rate']:.6f}"
        )

    # ---------------------------
    # Per-class metrics
    # ---------------------------
    if per_class:
        logger.comment("\nPer-class metrics:")

        for cls, m in metrics["per_class"].items():
            logger.comment(
                f"  [{cls}] "
                f"P={m['precision']:.4f} | "
                f"R={m['recall']:.4f} | "
                f"F1={m['f1']:.4f} | "
                f"FPR={m['fpr']:.6f} | "
                f"FNR={m['fnr']:.6f}"
            )


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