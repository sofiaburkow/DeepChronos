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
from src.evaluation.plots import plot_confusion_matrix


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
    stable_benign_scores = train_scores[int(len(train_scores) * 0.5):]
    
    # Fit the stable benign scores to a log-normal distribution
    log_scores = np.log(stable_benign_scores)
    mean_benign = np.mean(log_scores)
    std_benign = np.std(log_scores)

    log_threshold = norm.ppf(1 - alpha, mean_benign, std_benign)
    dynamic_threshold = np.exp(log_threshold)

    return dynamic_threshold


def main(
        data_dir, 
        out_dir,
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
    train_scores = []
    for i in range(total_train_rows):
        x = np.asarray(X_train[i], dtype=float)
        score = detector.process(x) 
        detector.process(x) 
        if (i+1) % 10000 == 0:
            print(f"Trained on {i+1} / {total_train_rows}")

        if i >= FM_grace:
            train_scores.append(score)
        if (i+1) % 10000 == 0:
            print(f"Trained on {i+1} / {total_train_rows}")

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
    cm = confusion_matrix(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred, cm, classes=["benign", "anomaly"], layout="actual_pred")
    metrics["threshold"] = dynamic_threshold
    misclassified = misclassified_samples(y_true, y_pred, y_test, num_phases=5)  # assuming phases 0-4
    metrics["missclassified"] = misclassified
    metrics["True Phase Distribution"] = Counter(y_test.tolist())
    
    print("Confusion matrix (tn, fp, fn, tp):")
    print(cm.ravel())
    print("Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    for phase, count in misclassified["per_phase"].items():
        print(f"Phase {phase}: {count} misclassified samples")

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_metrics_json(cm, metrics, out_file=out_dir/f"metrics_{run_id}.json")
    plot_confusion_matrix(cm, classes=["benign", "anomaly"], out_path=out_dir/f"confusion_matrix_{run_id}.png")



if __name__ == "__main__":
    # Command: uv run python -m Kitsune.experiment
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aitv2")
    parser.add_argument("--scenario", type=str, default="santos")
    args = parser.parse_args()

    data_dir = Path("data/kitsune/") / args.dataset / args.scenario
    out_dir = Path("experiments/") / args.dataset / args.scenario / "kitsune"
    main(data_dir=data_dir, out_dir=out_dir)