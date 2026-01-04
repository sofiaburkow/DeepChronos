import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.utils.class_weight import compute_class_weight

from network import FlowLSTM
from data import LSTMDataset
from helper_func import load_datasets, evaluate, misclassified_samples


def train_lstm(phase, dataset_dir, output_dir, batch_size=64, epochs=20):
    X_train, y_train, y_phases_train, X_test, y_test, y_phases_test = load_datasets(
        dataset_dir, phase, sparse=False
    )

    train_ds = LSTMDataset(X_train, y_train)
    test_ds  = LSTMDataset(X_test, y_test)

    # Sanity check
    print("Train samples:", len(train_ds))
    print("Test samples:", len(test_ds))

    x0, y0 = train_ds[0]
    print("One sample X shape:", x0.shape)
    print("One sample y:", y0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    input_dim = x0.shape[-1]
    print("Input dimension:", input_dim)

    model = FlowLSTM(input_dim)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.astype(int)
    )
    pos_weight = torch.tensor(class_weights[1] / class_weights[0])
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses = []

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)  # <-- record average loss
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    # ---- Evaluation ----
    acc, precision, recall, f1, cm, y_pred = evaluate(model, test_loader)

    print("\n=== Evaluation (Test set) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    misclassified_info = misclassified_samples(y_test, y_pred, y_phases_test)

    # ---- Save model + metrics ----
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"phase_{phase}.pth"
    torch.save(model.state_dict(), model_path)
    print("Saved pretrained LSTM to:", model_path)

    results_dir = model_dir / f"results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"phase_{phase}.json"
    with open(metrics_path, "w") as fh:
        json.dump(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
                "misclassified_info": misclassified_info
            },
            fh,
            indent=2,
        )
    print("Saved metrics to:", metrics_path)

    # ---- Save the loss plot ----
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    figure_path = results_dir / f"phase_{phase}_training_loss.png" 
    plt.savefig(figure_path, dpi=300)  # dpi=300 for high-quality
    plt.close()  # close the figure so it doesn't display in interactive sessions
    print(f"Training loss plot saved to: {figure_path}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="src/DARPA/data/processed")
    ap.add_argument("--out", default="src/DARPA/models/pretrained")
    args = ap.parse_args()

    for phase in range(1, 6):
        print(f"--- Training Model for Phase {phase} ---")
        train_lstm(phase, args.dataset_dir, args.out)


if __name__ == "__main__":
    # Command: uv run python src/DARPA/models/pretrained/create_pretrained.py
    cli()
