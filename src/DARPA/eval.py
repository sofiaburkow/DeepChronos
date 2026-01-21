from data.dataset import SubsetDPLDataset


def create_results_dirs(root_dir, run_id):
    """ Create results directories for snapshots and logs. """
    RESULTS_DIR = root_dir / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    snapshot_dir = RESULTS_DIR / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    log_dir = RESULTS_DIR / "logs" / run_id[:8]
    log_dir.mkdir(exist_ok=True)

    return snapshot_dir, log_dir


def all_phases_indices(dataset):
    """ Identify indices in the original dataset where all prev-phase flags are 1. """
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
    Currently supports 'all_phases' filter.
    """
    if filter_name == "all_phases":
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
    Compute TN/FP/FN/TP and standard metrics from a deepproblog ConfusionMatrix.

    Assumes a binary problem (2x2 matrix). 
    For the positive class we use the last entry in cm.classes.
    """

    # Use the underlying matrix and classes directly
    mat = getattr(cm, "matrix", None)
    classes = list(getattr(cm, "classes", []))

    if mat is None or len(classes) != 2:
        print("Confusion matrix is None or not binary; cannot compute metrics.")
        return None

    pos_label = classes[-1]
    pos_i = classes.index(pos_label)
    neg_i = 1 - pos_i

    # matrix[predicted_index, actual_index]
    TN = int(mat[neg_i, neg_i])
    FP = int(mat[pos_i, neg_i])
    FN = int(mat[neg_i, pos_i])
    TP = int(mat[pos_i, pos_i])

    total = TN + FP + FN + TP
    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0

    return {
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "positive_class": pos_label,
        "classes": classes,
    }