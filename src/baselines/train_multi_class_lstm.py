from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight

from src.networks.flow_lstm import LSTMClassifier
from src.datasets.flow_datasets import (
    load_windowed_data,
    WindowedFlowDataset
)
from src.evaluation.metrics import (
    evaluate, 
    save_metrics,
)
from src.evaluation.plots import save_loss_plot

    
def train_multi_class_lstm(
    processed_dir: Path,
    experiment_dir: Path,
    window_size: int,
    class_weights: bool,
    resampled: bool,
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",
):

    dataset_variant = "resampled" if resampled else "original"
    window_tag = f"w{window_size}"

    experiment_name = (
        "multi_class_lstm_"
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

    # ---- Datasets & Loaders ----
    train_ds = WindowedFlowDataset(data['train'], labels['train'])
    test_ds = WindowedFlowDataset(data['test'], labels['test'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ---- Build Model ----
    input_dim = train_ds[0][0].shape[-1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=6,  # benign + phases 1-5
        with_softmax=False,
    ).to(device)

    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---- Class Weights ----
    if class_weights:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2, 3, 4, 5]),
            y=labels['train'].astype(int)
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = CrossEntropyLoss()

    # ---- Training ----
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)  
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # ---- Evaluation ----
    acc, precision, recall, f1, cm, y_pred = evaluate(model, test_loader)

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
    )

    save_loss_plot(
        train_losses,
        epochs,
        out_file=results_dir / f"{experiment_name}_training_loss.png",
    )

    print("Saved model and metrics.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--resampled", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processed_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/windowed")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/baselines/multi_class_lstm")

    train_multi_class_lstm(
        processed_dir=processed_dir,
        experiment_dir=experiment_dir,
        window_size=args.window_size,
        class_weights=args.class_weights,
        resampled=args.resampled,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )
