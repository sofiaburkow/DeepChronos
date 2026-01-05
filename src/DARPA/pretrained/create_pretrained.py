"""
Train and save pretrained LSTM models for each DARPA phase.
"""

# Ensure src/DARPA is in sys.path (when running as a script)
import sys
from pathlib import Path
directory = Path(__file__).absolute().parent.parent
sys.path.append(str(directory))

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from sklearn.utils.class_weight import compute_class_weight

from network import FlowLSTM
from data.dataset import DARPAWindowedDataset
from pretrained.helper_func import (
    load_datasets, evaluate, misclassified_samples, save_metrics, save_loss_plot
)


def eval_lstm(model, test_loader, y_test, y_phases_test):
    """
    Evaluate the LSTM model on the test dataset and print metrics.
    """
    acc, precision, recall, f1, cm, y_pred = evaluate(model, test_loader)
    print("--- Evaluation (test set) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    misclassified_info = misclassified_samples(y_test, y_pred, y_phases_test)
    print(f"\nTotal misclassified samples: {misclassified_info['total_misclassified']}")
    for phase, count in misclassified_info["per_phase"].items():
        print(f"Phase {phase}: {count}")
    print()

    return acc, precision, recall, f1, cm, misclassified_info

    
def train_lstm(phase, dataset_dir, output_dir, batch_size=64, epochs=20):
    """
    Train and save a pretrained LSTM model for a specific DARPA phase.

    :param phase: Attack phase (1-5) to train the model for
    :param dataset_dir: Directory containing the processed dataset files
    :param output_dir: Directory to save the pretrained model and results
    :param batch_size: Batch size for training
    :param epochs: Number of training epochs
    """
    # ---- Load Data ----
    X_train, y_train, _, X_test, y_test, y_phases_test = load_datasets(
        dataset_dir=dataset_dir, 
        phase=phase, 
        sparse=False
    )

    # ---- Prepare DataLoaders ----
    train_ds = DARPAWindowedDataset(X_train, y_train)
    test_ds  = DARPAWindowedDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    # ---- Build Model ----
    input_dim = train_ds[0][0].shape[-1]
    model = FlowLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=2,
        with_softmax=False,
    )
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---- Compute Class Weights ----
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.astype(int)
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = CrossEntropyLoss(weight=class_weights_tensor)

    # ---- Training ----
    model.train()

    train_losses = []
    print("\nStarting training...")
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
    print("Training completed.\n")

    # ---- Evaluation ----
    acc, precision, recall, f1, cm, misclassified_info = eval_lstm(
        model, test_loader, y_test, y_phases_test
    )

    # ---- Save model + metrics ----
    pretrained_dir = Path(output_dir)
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = pretrained_dir / f"phase_{phase}.pth"
    torch.save(model.state_dict(), model_file)
    print("Saved pretrained LSTM to:", model_file)

    # Save metrics and loss plot
    results_dir = pretrained_dir / f"results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(phase, acc, precision, recall, f1, cm, misclassified_info, results_dir)
    save_loss_plot(train_losses, phase, epochs, results_dir)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="src/DARPA/data/processed")
    ap.add_argument("--out", default="src/DARPA/pretrained")
    args = ap.parse_args()

    # Train and save LSTM models per phase
    for phase in range(1, 6):
        print(f"\n=== Training Model for Phase {phase} ===")
        train_lstm(phase, args.dataset_dir, args.out)


if __name__ == "__main__":
    # Command: uv run python src/DARPA/pretrained/create_pretrained.py
    cli()
