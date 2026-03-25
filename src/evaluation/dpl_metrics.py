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

    for i, cls in enumerate(classes):

        TP = mat[i, i]
        FP = mat[i, :].sum() - TP
        FN = mat[:, i].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        support = mat[:, i].sum()

        per_class[cls] = {
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

    # Aggregates
    attack_classes = [c for c in classes if c != "benign"]

    precisions = np.array([per_class[c]["precision"] for c in attack_classes])
    recalls = np.array([per_class[c]["recall"] for c in attack_classes])
    f1s = np.array([per_class[c]["f1"] for c in attack_classes])
    supports = np.array([per_class[c]["support"] for c in attack_classes])

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports))

    # IDS metrics
    benign_idx = classes.index("benign")
    total_benign = mat[:, benign_idx].sum()
    benign_correct = mat[benign_idx, benign_idx]
    false_alarms = total_benign - benign_correct
    false_alarm_rate = false_alarms / total_benign if total_benign else 0.0

    # ---------------------------
    return dict(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        false_alarms=int(false_alarms),
        false_alarm_rate=false_alarm_rate,
        classes=classes,
        per_class=per_class,
    )


def log_metrics(logger, title, metrics, per_class):

    logger.comment(f"=== {title} ===")

    logger.comment("\nMetrics:")
    logger.comment(
        f"  Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro Precision: {metrics['macro_precision']:.4f} | "
        f"Macro Recall: {metrics['macro_recall']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )
    logger.comment(
        f"  Number of False Alarms: {metrics['false_alarms']} | "
        f"False Alarm Rate: {metrics['false_alarm_rate']:.4f}"
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