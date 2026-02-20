from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from src.networks.flow_lstm import EnsembleLSTMClassifier
from src.datasets.flow_datasets import (
    load_windowed_data,
    WindowedFlowDataset,
)
from src.evaluation.metrics import (
    evaluate, 
    save_metrics,
)
from src.evaluation.plots import save_loss_plot


def train_lstm(
        processed_dir: Path,
        experiment_dir: Path,
        window_size: int,
        num_classes: int,
        class_weights: bool,
        resampled: bool,
        batch_size: int, 
        epochs: int,
        device: str = "cpu",
):

    dataset_variant = "resampled" if resampled else "original"
    window_tag = f"w{window_size}"

    experiment_name = (
        "ensemble_lstm_"
        f"{dataset_variant}_"
        f"{'class_weights' if class_weights else 'no_class_weights'}_"
        f"{window_tag}"
    )

    print(f"\n=== Running {experiment_name} ===")

    # --- Load Datasets ---
    data, labels = load_windowed_data(
        base_dir=processed_dir,
        window_size=window_size,
        variant=dataset_variant,
    ) 

    # ---- Prepare DataLoaders ----
    train_dataset = WindowedFlowDataset(data['train'], labels['train'])
    test_dataset  = WindowedFlowDataset(data['test'], labels['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    
    # ---- Build Model ----
    input_dim = train_dataset[0][0].shape[-1]
    model = EnsembleLSTMClassifier(
        input_dim=input_dim, 
        hidden_dim=64, 
        output_dim=num_classes # one classifier per attack phase
    ).to(device)

    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---- Class Weights ----
    if class_weights:
        y_train_oh = torch.nn.functional.one_hot(
            torch.tensor(labels['train']), num_classes=num_classes
        ).float().to(device)
        pos_counts = y_train_oh.sum(axis=0)
        neg_counts = y_train_oh.shape[0] - pos_counts
        pos_weight = neg_counts / pos_counts
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=next(model.parameters()).device)
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = BCEWithLogitsLoss()

    # ---- Training ----
    model.train()

    train_losses = []
    print("\nStarting training...")
    for epoch in range(epochs):
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = torch.nn.functional.one_hot(y_batch, num_classes=num_classes).float().to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss) 
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
    
    # ---- Evaluation ----
    acc, precision, recall, f1, cm, y_pred = evaluate(model, test_loader)

    misclassified_indices = np.where(labels['test'] != y_pred)[0]
    t_test = np.load(processed_dir / f"w{window_size}" / dataset_variant / "t_test.npy")
    mis_t_indices = t_test[misclassified_indices]
    real_flow_indices = mis_t_indices + window_size - 1

    # ---- Save Artifacts ----
    model_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_dir / f"{experiment_name}.pth")

    save_metrics(
        acc, 
        precision, 
        recall, 
        f1, 
        cm, 
        out_file=results_dir / f"{experiment_name}_metrics.json",
        misclassified_indices=misclassified_indices,
        real_flow_indices=real_flow_indices
    )

    save_loss_plot(
        train_losses, 
        epochs, 
        out_file=results_dir / f"{experiment_name}_training_loss.png"
    )

    print("Saved model and metrics.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--resampled", action="store_true")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processed_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/windowed")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/baselines/ensemble_lstm")

    train_lstm(
        processed_dir=processed_dir, 
        experiment_dir=experiment_dir,
        window_size=args.window_size,
        num_classes=args.num_classes,
        class_weights=args.class_weights,
        resampled=args.resampled, 
        batch_size=args.batch_size, 
        epochs=args.epochs,
        device=device,
    )