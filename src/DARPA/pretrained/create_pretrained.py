import sys
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from sklearn.utils.class_weight import compute_class_weight

# Ensure src/DARPA is in sys.path (when running as a script)
directory = Path(__file__).absolute().parent.parent
sys.path.append(str(directory))

from network import FlowLSTM
from data.dataset import DARPAWindowedDataset
from eval import (
    evaluate, misclassified_samples, save_per_phase_metrics, save_loss_plot
)


def create_per_phase_labels(y_multi_class, phase):
    """
    Create binary labels for a specific attack phase.
    
    :param y_multi_class: Array of multi-class labels (1-5)
    :param phase: Attack phase (1-5) to create binary labels for
    :return: Binary labels (1 for samples in the specified phase, 0 otherwise)
    """
    return np.array([1 if label == phase else 0 for label in y_multi_class], dtype=np.int64)


def load_data(dataset_dir, phase):
    """
    Load DARPA datasets and create labels for a specific attack phase.
    
    :param dataset_dir: Directory containing the dataset files
    :param phase: Attack phase (1-5)
    """

    # Load features
    X_train = np.load(f"{dataset_dir}/X_train.npy", allow_pickle=True)
    X_test = np.load(f"{dataset_dir}/X_test.npy", allow_pickle=True)

    # Load labels
    y_train_multi_class = np.load(f"{dataset_dir}/y_train_multi_class.npy", allow_pickle=True)
    y_test_multi_class = np.load(f"{dataset_dir}/y_test_multi_class.npy", allow_pickle=True) 

    # Create binary labels for the specified phase
    y_train_per_phase = create_per_phase_labels(y_train_multi_class, phase)
    y_test_per_phase = create_per_phase_labels(y_test_multi_class, phase)

    return X_train, X_test, y_train_multi_class, y_test_multi_class, y_train_per_phase, y_test_per_phase

    
def train_lstm(phase, dataset_dir, output_dir, window_size, resampled, batch_size=64, epochs=20):
    """
    Train and save a pretrained LSTM model for a specific DARPA phase.

    :param phase: Attack phase (1-5) to train the model for
    :param dataset_dir: Directory containing the processed dataset files
    :param output_dir: Directory to save the pretrained model and results
    :param window_size: Size of the time window for the features
    :param resampled: Whether to use the resampled dataset or original
    :param batch_size: Batch size for training
    :param epochs: Number of training epochs
    """

    config = f"w{window_size}/" +(f"resampled" if resampled else "original")

    # ---- Load Data ----
    X_train, X_test, _, y_test_multi_class, y_train_per_phase, y_test_per_phase = load_data(
        dataset_dir=Path(dataset_dir) / config,
        phase=phase
    )

    # ---- Prepare DataLoaders ----
    train_ds = DARPAWindowedDataset(X_train, y_train_per_phase)
    test_ds  = DARPAWindowedDataset(X_test, y_test_per_phase)

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
        y=y_train_per_phase.astype(int)
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
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
    print("Training completed.\n")

    # ---- Evaluation ----
    acc, precision, recall, f1, cm, y_pred = evaluate(model, test_loader)
    misclassified_info = misclassified_samples(y_test_per_phase, y_pred, y_test_multi_class)

    # ---- Save model ----
    models_dir = Path(output_dir) / "models" / config
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"phase_{phase}.pth"

    torch.save(model.state_dict(), model_file)
    print("Saved pretrained LSTM to:", model_file)

    # ---- Save metrics and loss plot ----
    results_dir = Path(output_dir) / "results" / config
    results_dir.mkdir(parents=True, exist_ok=True)

    save_per_phase_metrics(
        acc, precision, recall, f1, cm, misclassified_info, 
        out_file= results_dir / f"phase_{phase}_metrics.json"
    )

    save_loss_plot(
        train_losses, epochs, 
        out_file= results_dir / f"phase_{phase}_training_loss.png"
    )


if __name__ == "__main__":
    # Command: uv run python src/DARPA/pretrained/create_pretrained.py --window_size 50 --resampled

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="src/DARPA/data/processed")
    ap.add_argument("--output_dir", default="src/DARPA/pretrained", help="Output directory to save the trained model and results")
    ap.add_argument("--window_size", type=int, default=10, help="Size of the time window for the features")
    ap.add_argument("--resampled", action="store_true", default=True, help="Whether to use resampled dataset")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train and save LSTM models per phase
    for phase in range(1,6):
        print(f"\n=== Training Model for Phase {phase} ===")
        train_lstm(
            phase=phase, 
            dataset_dir=args.dataset_dir, 
            output_dir=args.output_dir,
            window_size=args.window_size,
            resampled=args.resampled,
            batch_size=args.batch_size, 
            epochs=args.epochs
        )
