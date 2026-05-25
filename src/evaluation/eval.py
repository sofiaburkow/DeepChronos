from collections import Counter
import time 
from typing import List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from deepproblog.dataset import Dataset
from deepproblog.model import Model
from deepproblog.utils.confusion_matrix import ConfusionMatrix


def eval_lstm(
    model,
    dataloader,
    multiclass: bool = True,
    device: torch.device | None = None,
):
    """
    Evaluate a PyTorch model on a given dataloader.

    :param model: The PyTorch model to evaluate.
    :param dataloader: DataLoader providing the evaluation data.
    :param multiclass: Whether the model is multi-class (True) or binary (False).
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
            if multiclass:
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
    if multiclass:
        classes = ["benign" if c == 0 else f"phase{c}" for c in classes]
    else:
        classes = ["benign", "attack"]

    return cm, classes, y_true, y_pred


def get_confusion_matrix(
    model: Model, 
    dataset: Dataset, 
    classes: List[str] = None,
    verbose: int = 0,
):
    """
    Evaluate a DeepProbLog model on a dataset and return the confusion matrix and misclassified examples.
    """
    confusion_matrix = ConfusionMatrix(classes=classes) if classes is not None else ConfusionMatrix()

    model.eval()

    inference_times = []
    misclassified = []
    correct = []
    y_true = []
    y_pred = []

    with torch.no_grad(): # do I need this?
        for i, gt_query in enumerate(dataset.to_queries()):
            test_query = gt_query.variable_output()

            start = time.time()
            answer = model.solve([test_query])[0]
            end = time.time()
            inference_time = end - start
            inference_times.append(inference_time)

            actual = str(gt_query.output_values()[0])

            if len(answer.result) == 0:
                predicted = "no_answer"
                p = None
            else:
                max_ans = max(answer.result, key=lambda x: answer.result[x])
                p = answer.result[max_ans]
                predicted = str(max_ans.args[gt_query.output_ind[0]])

            if actual != predicted:
                misclassified.append({
                    "index": i,
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": p,
                    "test_query": test_query,
                })

            else:
                correct.append({
                    "index": i,
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": p,
                    "test_query": test_query,
                })

            y_true.append(actual)
            y_pred.append(predicted)

            confusion_matrix.add_item(predicted, actual)

    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())
    
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time per query: {avg_inference_time:.4f} seconds")
    print(f"Total inference time: {sum(inference_times):.2f} seconds")

    return confusion_matrix, misclassified, correct, inference_times, y_true, y_pred
    

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

    num_phases = 5  # phases 0 = benign, 1-5 = attack phases
    phases = list(range(num_phases))

    return {
        "total_misclassified": len(misclassified_indices),
        "per_phase": {phase: int(counts.get(phase, 0)) for phase in phases}
    }


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

