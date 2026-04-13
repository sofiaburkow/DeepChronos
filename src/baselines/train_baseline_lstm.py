from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from src.networks.flow_lstm import EnsembleLSTMClassifier, LSTMClassifier
from src.datasets.flow_datasets import load_windowed_data, WindowedFlowDataset
from src.evaluation.lstm_metrics import eval, save_metrics
from src.evaluation.dpl_metrics import compute_metrics
from src.evaluation.plots import save_loss_plot, plot_confusion_matrix


def build_weighted_sampler(labels):
    """Build a weighted random sampler based on class frequencies."""

    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def train_lstm(
    classifier: str,
    data_dir: Path,
    experiment_dir: Path,
    feature_group: str,
    window_size: int,
    fraction: int,
    batch_size: int, 
    epochs: int,
    device: str = "cpu",
):

    experiment_name = (
        f"{classifier}_"
        f"{feature_group}features_"
        f"{fraction}data_"
        f"w{window_size}"
    )

    print(f"\n=== Running {experiment_name} ===")

    # --- Load Data ---
    data, labels, _, _ = load_windowed_data(data_dir=data_dir, fraction=fraction)
    train_dataset = WindowedFlowDataset(data['train'], labels['train'])
    test_dataset  = WindowedFlowDataset(data['test'], labels['test'])

    # Weighted sampler to handle class imbalance in training set
    sampler = build_weighted_sampler(train_dataset.y)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
    )

    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    
    # ---- Build Model ----
    num_classes = len(set(labels['train']))
    print(f"Number of classes: {num_classes}")
    input_dim = train_dataset[0][0].shape[-1]

    if classifier == "ensemble":
        model = EnsembleLSTMClassifier(
            input_dim=input_dim, 
            hidden_dim=64, 
            output_dim=num_classes
        ).to(device)
        criterion = BCEWithLogitsLoss()

    elif classifier == "multiclass":
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=num_classes,
            with_softmax=False,
        ).to(device)
        criterion = CrossEntropyLoss()
    
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---- Training ----
    model.train()
    train_losses = []

    print("\nStarting training...")
    for epoch in range(epochs):
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)

            if classifier == "ensemble":
                y_batch = torch.nn.functional.one_hot(y_batch, num_classes=num_classes).float().to(device)
            elif classifier == "multiclass":
                y_batch = y_batch.to(device)

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
    cm, classes, y_pred = eval(model, test_loader, multiclass=True, device=device)
    metrics = compute_metrics(cm, classes, layout="actual_pred")

    # Analyze misclassifications
    t_test = np.load(data_dir / "t_test.npy")
    misclassified_indices = np.where(labels['test'] != y_pred)[0]
    mis_t_indices = t_test[misclassified_indices]
    real_flow_indices = mis_t_indices + window_size - 1
    mis_y_pred = y_pred[misclassified_indices]
    mis_y_true = labels['test'][misclassified_indices]

    # ---- Save Artifacts ----
    model_dir = experiment_dir / "models"
    metrics_dir = experiment_dir / "metrics"
    plots_dir = experiment_dir / "plots"

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / f"{experiment_name}.pth")

    save_metrics(
        cm, 
        metrics,
        mis_indices=misclassified_indices,
        real_indices=real_flow_indices,
        y_pred=mis_y_pred,
        y_true=mis_y_true,
        out_file=metrics_dir / f"{experiment_name}_metrics.json",
    )

    save_loss_plot(
        train_losses, 
        epochs, 
        out_file=plots_dir / f"{experiment_name}_training_loss.png"
    )
    
    plot_confusion_matrix(
        cm=np.array(cm).T, # Transpose to get actual vs predicted layout
        classes=classes, 
        experiment_name=experiment_name,
        out_path = plots_dir / f"{experiment_name}_cm.png",
    )


if __name__ == "__main__":
    # uv run python -m src.baselines.train_baseline_lstm --classifier multiclass --dataset aitv2 --scenario fox --feature_group all --window_size 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--classifier", type=str, default="ensemble", choices=["ensemble", "multiclass"])
    parser.add_argument("--feature_group", type=str, default="all")
    parser.add_argument("--fraction", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/{args.feature_group}/windowed/w{args.window_size}")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/baselines/{args.classifier}")

    train_lstm(
        classifier=args.classifier, 
        data_dir=data_dir, 
        experiment_dir=experiment_dir,
        feature_group=args.feature_group,  
        window_size=args.window_size,
        fraction=args.fraction,
        batch_size=args.batch_size, 
        epochs=args.epochs,
        device=device,
    )