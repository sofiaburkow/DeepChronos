import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

from Kitsune.KitNET.KitNET import KitNET

from sklearn.metrics import confusion_matrix
from src.evaluation.eval import misclassified_samples   
from src.evaluation.metrics import compute_metrics, save_metrics_json
from src.evaluation.plots import plot_confusion_matrix, plot_confusion_matrix_2x5


def load_matrix(p):
    if p.with_suffix(".npz").exists():
        mat = sp.load_npz(p.with_suffix(".npz"))
        return mat  # sparse
    if p.exists():
        return np.load(p, mmap_mode="r")
    raise FileNotFoundError(p)


def load_data(data_dir):
    X_train_npy = data_dir / "X_train.npy"
    X_test_npy  = data_dir / "X_test.npy"
    y_test_npy  = data_dir / "y_test.npy"

    X_train = load_matrix(X_train_npy)
    X_test = load_matrix(X_test_npy)
    y_test = np.load(y_test_npy)

    # If train is sparse, convert if memory permits
    if sp.issparse(X_train):
        X_train = X_train.toarray()

    # If X_test is sparse, convert to dense rows when iterating (or toarray() if small)
    if sp.issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    
    return X_train, X_test_dense, y_test


def compute_dynamic_threshold(train_scores, alpha=0.01):
    # Take the second half of the AD_grace scores where the model is most stable
    stable_benign_scores = np.array(train_scores[int(len(train_scores) * 0.5):])

    # Filter out non-positive or non-finite scores which would break the log
    stable_benign_scores = stable_benign_scores[np.isfinite(stable_benign_scores) & (stable_benign_scores > 0)]

    if stable_benign_scores.size == 0:
        # Fallback: use a high percentile of the (raw) train_scores if no valid positives found
        if len(train_scores) == 0:
            raise ValueError("No training scores available to compute a threshold")
        return float(np.percentile(np.array(train_scores[np.isfinite(train_scores)]), 100 * (1 - alpha)))

    # Fit the stable benign scores to a log-normal distribution
    log_scores = np.log(stable_benign_scores)
    mean_benign = np.mean(log_scores)
    std_benign = np.std(log_scores)

    log_threshold = norm.ppf(1 - alpha, mean_benign, std_benign)
    dynamic_threshold = float(np.exp(log_threshold))

    return dynamic_threshold


def main(
        data_dir, 
        out_dir,
        classes,
        true_classes,
        m=10,
    ):

    # Load data
    X_train, X_test, y_test = load_data(data_dir)
    n_features = int(X_train.shape[1])
    total_train_rows = X_train.shape[0]
    
    # Initialize KitNET
    FM_grace = int(total_train_rows * 0.2)  # Use first X% of training data for feature mapping
    AD_grace = total_train_rows - FM_grace
    detector = KitNET(
        n_features, 
        max_autoencoder_size=m, 
        FM_grace_period=FM_grace, 
        AD_grace_period=AD_grace, 
        learning_rate=0.1, 
        hidden_ratio=0.75
    )
    
    print(f"Replaying training data (FM_grace={FM_grace}, AD_grace={AD_grace})...")
    # Single-pass training: call process() once per sample to advance KitNET's internal state
    for i in range(total_train_rows):
        x = np.asarray(X_train[i], dtype=float)
        _ = detector.process(x)
        if (i+1) % 10000 == 0:
            print(f"Trained on {i+1} / {total_train_rows}")

    # After training completes, collect the training (benign) scores by running in execute-mode
    # This avoids relying on a duplicated process() call earlier which effectively accelerated
    # the grace counters and produced execute-mode outputs.
    train_scores = []
    for i in range(total_train_rows):
        if i < FM_grace:
            # skip FM-only period samples
            continue
        x = np.asarray(X_train[i], dtype=float)
        s = detector.execute(x)
        train_scores.append(s)
        if (i+1) % 10000 == 0:
            print(f"Scored {i+1} / {total_train_rows} training rows")

    # Compute dynamic threshold based on training scores
    print("Calibrating threshold...")
    alpha = 0.01  # Desired false positive rate (1%)
    dynamic_threshold = compute_dynamic_threshold(train_scores, alpha)
    print(f"Dynamically calculated threshold at {alpha*100}% FPR: {dynamic_threshold}")

    # Test: strictly in execution mode for this entire loop
    print("Running on test set (Execution mode)...")
    test_scores = []
    for i in range(X_test.shape[0]):
        x = np.asarray(X_test[i], dtype=float)
        s = detector.process(x)  
        test_scores.append(s)
    test_scores = np.array(test_scores)

    # Compute metrics
    y_pred = (test_scores > dynamic_threshold).astype(int)  # 1 => anomaly
    y_true = (y_test != 0).astype(int)              # assuming phase 0 = benign, non-zero = anomaly

    cm_binary = confusion_matrix(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred, cm_binary, classes=["benign", "anomaly"], layout="actual_pred")
    metrics["threshold"] = dynamic_threshold
    misclassified = misclassified_samples(y_true, y_pred, y_test, num_phases=5)  # assuming phases 0-4
    metrics["missclassified"] = misclassified
    metrics["True Phase Distribution"] = Counter(y_test.tolist())

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_metrics_json(cm_binary, metrics, out_file=out_dir/f"metrics_{run_id}.json")
    plot_confusion_matrix(cm_binary, classes=["benign", "anomaly"], out_path=out_dir/f"confusion_matrix_binary_{run_id}.png")
    plot_confusion_matrix_2x5(y_pred, y_test, classes, true_classes, ["benign", "anomaly"], out_path=out_dir/f"confusion_matrix_2x5_{run_id}.png")


if __name__ == "__main__":
    # Command: uv run python -m Kitsune.msa_experiment

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aitv2")
    parser.add_argument("--scenario", type=str, default="santos")
    args = parser.parse_args()

    dataset = args.dataset
    scenario = args.scenario
    
    # options = {
    #     ("darpa2000", "s1_inside"),
    #     ("darpa2000", "s1_dmz"),
    #     ("aitv2", "santos"),
    #     ("aitv2", "fox"),
    # }

    # for (dataset, scenario) in options:
    print(f"\n=== Running KitNET on {dataset} - {scenario} ===")
    data_dir = Path("data/kitsune/") / dataset / scenario
    out_dir = Path("experiments/") / dataset / scenario / "kitsune"

    if dataset == "darpa2000":
        classes = [0, 1, 2, 3, 4, 5]
        true_classes = ['Benign', 'Phase1', 'Phase2', 'Phase3', 'Phase4', "Phase5"]
    elif dataset == "aitv2":
        classes = [0, 1, 2, 3, 4]
        true_classes = ['Benign', 'Phase1', 'Phase2', 'Phase3', 'Phase4']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    main(
        data_dir=data_dir, 
        out_dir=out_dir,
        classes=classes,
        true_classes=true_classes,
    )