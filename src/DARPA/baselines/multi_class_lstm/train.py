import sys
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from sklearn.utils.class_weight import compute_class_weight

# Ensure src/DARPA is in sys.path (when running as a script)
directory = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(directory))

from network import FlowLSTM
from data.dataset import DARPAWindowedDataset
from eval import evaluate, save_metrics, save_confusion_matrix_heatmap, save_loss_plot
from helper_func import load_data

    
def train_lstm(output_dir, resampled_bool, class_weights_bool, batch_size=64, epochs=20):
    """
    Train and save a DARPA multi-class (benign + phase 1-5) LSTM model.

    :param output_dir: Directory to save the trained model and results
    :param resampled_bool: Whether to train on the resampled dataset
    :param class_weights_bool: Whether to use class weights during training
    :param batch_size: Batch size for training
    :param epochs: Number of training epochs
    """

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    resampled_str = "resampled" if resampled_bool else "original"
    class_weights_str = "class_weights" if class_weights_bool else "no_class_weights"

    # ---- Load Data ----
    datasets_data, datasets_labels = load_data(resampled_str)
    X_train, y_train = datasets_data["train"], datasets_labels["train"]
    X_test, y_test = datasets_data["test"], datasets_labels["test"]

    # ---- Prepare DataLoaders ----
    train_dataset = DARPAWindowedDataset(X_train, y_train)
    test_dataset  = DARPAWindowedDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    
    # ---- Build Model ----
    input_dim = train_dataset[0][0].shape[-1]
    model = FlowLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=6,  # 6 classes: benign + phases 1-5
        with_softmax=False,
    )
    learning_rate = 1e-4 # same lr as in DPL model
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---- Loss Function ----
    if class_weights_bool:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2, 3, 4, 5]),
            y=y_train.astype(int)
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        criterion = CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = CrossEntropyLoss()

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

    output_dir = Path(output_dir) / resampled_str / class_weights_str / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / f"model.pth"
    torch.save(model.state_dict(), model_file)
    print("Saving trained multi-class LSTM model to:", model_file)

    # ---- Evaluation ----
    print("\nEvaluating model on test set...")
    acc, precision, recall, f1, cm, _ = evaluate(model, test_loader)
    print(f"Confusion Matrix:\n{cm}")
   
    save_metrics(
        acc, precision, recall, f1, cm, 
        out_file = output_dir / f"metrics.json"
    )
    save_confusion_matrix_heatmap(
        cm, 
        class_names=["Benign", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"],
        title = "Multi-Class LSTM Confusion Matrix",
        out_file = output_dir / f"confusion_matrix.png",
    )
    save_loss_plot(
        train_losses, epochs, 
        out_file = output_dir / f"training_loss.png"
    )


if __name__ == "__main__":
    # Command: uv run python src/DARPA/baselines/multi_class_lstm/train.py --resampled

    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="src/DARPA/baselines/multi_class_lstm", help="Output directory to save the trained model and results")
    ap.add_argument("--resampled", action="store_true", help="Whether to train on the resampled dataset")
    ap.add_argument("--class_weights", action="store_true", help="Whether to use class weights during training")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train multi-class LSTM model
    print(f"=== Training multi-class LSTM ===")
    print(f"Resampled dataset: {args.resampled}")
    print(f"Using class weights: {args.class_weights}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    train_lstm(
        output_dir=args.output_dir, 
        resampled_bool=args.resampled, 
        class_weights_bool=args.class_weights,
        batch_size=args.batch_size, 
        epochs=args.epochs
    )
