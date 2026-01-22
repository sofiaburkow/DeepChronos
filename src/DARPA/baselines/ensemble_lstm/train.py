import sys
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

# Ensure src/DARPA is in sys.path (when running as a script)
directory = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(directory))

from network import EnsembleLSTM
from data.dataset import DARPAWindowedDataset
from eval import evaluate_ensemble, save_metrics, save_confusion_matrix_heatmap, save_loss_plot


def load_data(resampled: bool):
    """Load DARPA datasets from disk."""

    resampled_str = "resampled" if resampled else "original"

    datasets_data = {
        "train": np.load(f"src/DARPA/data/processed/{resampled_str}/X_train.npy", allow_pickle=True),
        "test":  np.load(f"src/DARPA/data/processed/{resampled_str}/X_test.npy", allow_pickle=True),
    }
    datasets_labels = {
        "train": np.load(f"src/DARPA/data/processed/{resampled_str}/y_train_multi_class.npy", allow_pickle=True),
        "test":  np.load(f"src/DARPA/data/processed/{resampled_str}/y_test_multi_class.npy", allow_pickle=True),
    }

    return datasets_data, datasets_labels

    
def train_lstm(output_dir, resampled, batch_size=64, epochs=20):
    """
    Train and save a DARPA ensemble LSTM model.

    :param output_dir: Directory to save the trained model and results
    :param resampled: Whether to train on the resampled dataset
    :param batch_size: Batch size for training
    :param epochs: Number of training epochs
    """

    run_id = datetime.now().strftime("%Y%m%d_%H%M")

    # ---- Load Data ----
    datasets_data, datasets_labels = load_data(resampled=resampled)
    X_train, y_train = datasets_data["train"], datasets_labels["train"]
    X_test, y_test = datasets_data["test"], datasets_labels["test"]

    # ---- Prepare DataLoaders ----
    train_dataset = DARPAWindowedDataset(X_train, y_train)
    test_dataset  = DARPAWindowedDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    
    # ---- Build Model ----
    input_dim = train_dataset[0][0].shape[-1]
    model = EnsembleLSTM(
        input_dim=input_dim, 
        hidden_dim=64, 
        output_dim=6
    )
    learning_rate = 1e-3 # TODO: figure out best lr
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---- Compute Class Weights ----
    y_train_oh = np.eye(6)[y_train.astype(int)]
    pos_counts = y_train_oh.sum(axis=0)
    neg_counts = y_train_oh.shape[0] - pos_counts

    pos_weight = neg_counts / pos_counts
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # ---- Training ----
    model.train()

    train_losses = []
    print("\nStarting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            y_batch = torch.nn.functional.one_hot(y_batch, num_classes=6).float()
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)  # <-- record average loss
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    resampled_str = "resampled" if resampled else "original"
    output_dir = Path(output_dir) / resampled_str / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / f"model.pth"
    torch.save(model.state_dict(), model_file)
    print("Saving trained ensemble LSTM model to:", model_file)

    # ---- Evaluation ----
    print("\nEvaluating model on test set...")
    acc, precision, recall, f1, cm, _ = evaluate_ensemble(model, test_loader)
    missclassified_info = None  # Not needed for multi-class
   
    save_metrics(
        acc, precision, recall, f1, cm, missclassified_info, 
        out_file = output_dir / f"metrics.json"
    )
    save_confusion_matrix_heatmap(
        cm, 
        class_names=["Benign", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"],
        title = "Ensemble LSTM Confusion Matrix",
        out_file = output_dir / f"confusion_matrix.png",
    )
    save_loss_plot(
        train_losses, epochs, 
        out_file = output_dir / f"training_loss.png"
    )


if __name__ == "__main__":
    # Command: uv run python src/DARPA/baselines/ensemble_lstm/train.py

    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="src/DARPA/baselines/ensemble_lstm", help="Output directory to save the trained model and results")
    ap.add_argument("--resampled", action="store_true", help="Whether to train on the resampled dataset")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train ensemble LSTM model
    print(f"=== Training ensemble LSTM ===")
    print(f"Resampled dataset: {args.resampled}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    train_lstm(
        output_dir=args.output_dir, 
        resampled=args.resampled, 
        batch_size=args.batch_size, 
        epochs=args.epochs
    )
