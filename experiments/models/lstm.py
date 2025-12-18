"""Train a small LSTM on preprocessed datasets.

Usage:
    uv run python experiments/models/lstm.py <dataset_dir> <true|false> [--seq-len N]

This script tries to be robust to the project's preprocessing output (sparse X_train/X_test).
It will convert to dense, attempt to reshape into (samples, seq_len, feat_dim) where seq_len is
either provided via --seq-len or inferred (if features divisible by seq_len). If inference fails
the script falls back to seq_len=1 (a single timestep, equivalent to a dense model run through LSTM).

Dependencies: tensorflow (keras), scikit-learn, numpy, scipy
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

from helper_fun.train_func import load_datasets
from helper_fun.eval_func import plot_misclassified_samples


def to_dense_and_reshape(X, seq_len: int):
    """Convert sparse matrix to dense and reshape to (n_samples, seq_len, feat_dim).

    If number of features isn't divisible by seq_len, raise ValueError.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)

    n_samples, n_features = X.shape
    if seq_len <= 0:
        raise ValueError("seq_len must be >= 1")
    if n_features % seq_len != 0:
        raise ValueError(f"n_features ({n_features}) not divisible by seq_len ({seq_len})")
    feat_dim = n_features // seq_len
    Xr = X.reshape((n_samples, seq_len, feat_dim))
    return Xr


def build_lstm_model(input_shape, dropout=0.2):
    model = Sequential()
    # Stack layers
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # model.summary() # Debugging step

    return model


def train_and_evaluate(dataset_dir: str, sample_weights: bool, seq_len: int):

    X_train, y_train, y_phase_train, X_test, y_test, y_phase_test = load_datasets(dataset_dir)

    # Try to reshape using provided seq_len; if not possible, try to infer a reasonable seq_len
    tried_seq = seq_len
    X_train_r = None
    X_test_r = None
    n_features = X_train.shape[1]
    if seq_len is None:
        # heuristic: try some common sequence lengths
        for candidate in [10, 8, 5, 4, 2, 1]:
            if n_features % candidate == 0:
                tried_seq = candidate
                break
    try:
        X_train_r = to_dense_and_reshape(X_train, tried_seq)
        X_test_r = to_dense_and_reshape(X_test, tried_seq)
    except ValueError:
        # fallback: single timestep
        print(f"Could not reshape to seq_len={tried_seq}; falling back to seq_len=1")
        tried_seq = 1
        X_train_r = to_dense_and_reshape(X_train, tried_seq)
        X_test_r = to_dense_and_reshape(X_test, tried_seq)

    print(f"Using seq_len={tried_seq}; input shape for LSTM: {X_train_r.shape[1:]} (timesteps, features)")

    model = build_lstm_model(input_shape=X_train_r.shape[1:])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    sample_weight_arr = None
    if sample_weights:
        sample_weight_arr = compute_sample_weight(class_weight="balanced", y=y_train)

    history = model.fit(
        X_train_r,
        y_train,
        epochs=50,
        # epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=[es],
        verbose=2,
        sample_weight=sample_weight_arr,
    )

    # Predict and evaluate
    y_pred_prob = model.predict(X_test_r).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save model and artifacts
    parts = dataset_dir.rstrip("/").split("/")
    base_out = Path("lstm") / ("sample_weights" if sample_weights else "no_sample_weights") / "".join(parts[-3:])
    base_out.mkdir(parents=True, exist_ok=True)
    model_file = base_out / "model.h5"
    history_file = base_out / "history.json"
    metrics_file = base_out / "metrics.json"
    model.save(str(model_file))

    misclassified_samples_file = base_out / "misclassified_samples.png"
    plot_misclassified_samples(y_test, y_pred, y_phase_test, misclassified_samples_file)

    # Save history (json serialisable)
    with open(history_file, "w") as fh:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, fh)

    with open(metrics_file, "w") as fh:
        json.dump({"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1), "confusion_matrix": cm.tolist()}, fh)

    print("=== LSTM Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


def cli(argv=None):
    ap = argparse.ArgumentParser(description="Train an LSTM on processed dataset")
    ap.add_argument("dataset_dir")
    ap.add_argument("sample_weights", help="true|false")
    ap.add_argument("--seq-len", type=int, default=None, help="Number of timesteps to reshape features into (optional)")
    args = ap.parse_args(argv)

    dataset_dir = args.dataset_dir
    sample_weights = str(args.sample_weights).lower() == "true"
    seq_len = args.seq_len
    train_and_evaluate(dataset_dir, sample_weights, seq_len)


if __name__ == "__main__":
    # Command: uv run python experiments/models/lstm.py <dataset_dir> <true|false> [--seq-len N]
    cli()
