import numpy as np
import torch

from typing import Optional

from deepproblog.dataset import Dataset
from deepproblog.model import Model
from deepproblog.utils.confusion_matrix import ConfusionMatrix


def snapshot_params(net):
    """ 
    Take a snapshot of the parameters of a PyTorch module. 
    """
    return {n: p.detach().cpu().clone() for n, p in net.named_parameters()}


def print_param_changes(modules, snapshots_before):
    """ 
    Print parameter changes for monitored PyTorch modules. 
    """
    for i, net in enumerate(modules):
        before = snapshots_before[i]
        after = snapshot_params(net)  
        for name in before:
            diff = (after[name] - before[name]).norm().item()
            before_norm = before[name].norm().item() or 1e-12
            rel = diff / before_norm
            print(f"{name}: L2-change={diff:.6e}, relative={rel:.4%}")


def get_confusion_matrix(
    model: Model, dataset: Dataset, verbose: int = 0, eps: Optional[float] = None
):
    """
    Evaluate a DeepProbLog model on a dataset and return the confusion matrix and misclassified examples.
    """

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
    

def compute_metrics(confusion_matrix, classes, layout="actual_pred"):
    """
    Compute unified IDS metrics from a confusion matrix.

    Parameters
    ----------
    confusion_matrix : array-like (NxN)
        Confusion matrix.

    classes : list
        Class labels in matrix order.
        Must include a "benign" class.

    layout : str
        "actual_pred"   -> cm[actual, predicted]  (sklearn default)
        "pred_actual"   -> cm[predicted, actual]  (DeepProbLog)

    Returns
    -------
    dict containing:
        accuracy
        macro_precision
        macro_recall
        macro_f1
        false_alarm_rate
        detection_rate
        per_class metrics
    """

    cm = np.asarray(confusion_matrix, dtype=float)

    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    if layout not in {"actual_pred", "pred_actual"}:
        raise ValueError("layout must be 'actual_pred' or 'pred_actual'")

    # Orientation: [predicted, actual]
    if layout == "actual_pred":
        cm = cm.T

    classes = list(classes)
    if "benign" not in classes:
        raise ValueError("Classes must contain 'benign' label.")

    benign_idx = classes.index("benign")
    n_classes = len(classes)

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------
    total = cm.sum()
    accuracy = np.trace(cm) / total if total else 0.0

    # --------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------
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

        support = cm[:, i].sum()

        per_class[cls] = dict(
            TP=int(TP),
            FP=int(FP),
            FN=int(FN),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            support=int(support),
        )

    # --------------------------------------------------
    # Macro metrics
    # --------------------------------------------------

    precisions = np.array([per_class[c]["precision"] for c in classes])
    recalls = np.array([per_class[c]["recall"] for c in classes])
    f1s = np.array([per_class[c]["f1"] for c in classes])
    supports = np.array([per_class[c]["support"] for c in classes])

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports)) if supports.sum() else 0.0

    # --------------------------------------------------
    # IDS Metrics
    # --------------------------------------------------

    attack_idxs = [i for i in range(n_classes) if i != benign_idx]

    # ---- False Alarm Rate (FAR)
    # benign predicted as attack
    total_benign = cm[:, benign_idx].sum()
    benign_correct = cm[benign_idx, benign_idx]

    false_alarms = total_benign - benign_correct
    false_alarm_rate = false_alarms / total_benign if total_benign else 0.0

    # ---- Detection Rate (DR)
    # attacks correctly detected
    TP_attacks = sum(cm[i, i] for i in attack_idxs)

    FN_attacks = sum(
        cm[benign_idx, j]   # predicted benign while actual attack
        for j in attack_idxs
    )

    detection_rate = (
        TP_attacks / (TP_attacks + FN_attacks)
        if (TP_attacks + FN_attacks)
        else 0.0
    )

    # --------------------------------------------------
    return dict(
        accuracy=float(accuracy),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        false_alarms=int(false_alarms),
        false_alarm_rate=float(false_alarm_rate),
        missed_attacks=int(FN_attacks),
        detection_rate=float(detection_rate),
        classes=classes,
        per_class=per_class,
    )


def save_metrics(cm, metrics, out_path):

    mat, classes = extract_cm(cm)

    np.savez(
        out_path,
        confusion_matrix=mat,
        classes=np.array(classes),
        metrics=metrics
    )

    print(f"Saved metrics to: {out_path}")


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