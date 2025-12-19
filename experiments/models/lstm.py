import argparse
import json
from pathlib import Path

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

from helper_fun.train_func import load_datasets
from helper_fun.eval_func import plot_misclassified_samples


def build_lstm_model(input_shape, dropout=0.2):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_and_evaluate(dataset_dir: str, sample_weights: bool, seq_len: int):

    X_train, y_train, y_phase_train, X_test, y_test, y_phase_test = load_datasets(dataset_dir, sparse=False)

    model = build_lstm_model(input_shape=X_train.shape[1:])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    sample_weight_arr = None
    if sample_weights:
        sample_weight_arr = compute_sample_weight(class_weight="balanced", y=y_train)

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[es],
        verbose=2,
        sample_weight=sample_weight_arr,
    )

    # Predict and evaluate
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save model and artifacts
    parts = dataset_dir.rstrip("/").split("/")
    base_out = Path("experiments/results/lstm/") / ("sample_weights" if sample_weights else "no_sample_weights") / parts[-2] / parts[-1]

    model_file = base_out / "model.h5"
    history_file = base_out / "history.json"
    metrics_file = base_out / "metrics.json"
    model.save(str(model_file))

    plot_misclassified_samples(y_test, y_pred, y_phase_test, base_out)

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
