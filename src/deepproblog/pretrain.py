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
    misclassified_samples,
    save_per_phase_metrics,
)
from src.evaluation.plots import save_loss_plot


def create_per_phase_labels(y_multi_class: np.ndarray, phase: int) -> np.ndarray:
    """
    Convert multi-class labels into binary labels for a specific phase.
    """
    return (y_multi_class == phase).astype(np.int64)


def train_phase_classifier(
    phase: int,
    processed_dir: Path,
    experiment_dir: Path,
    window_size: int,
    resampled: bool,
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",
):

    dataset_variant = "resampled" if resampled else "original"
    dataset_path = processed_dir / f"w{window_size}" / dataset_variant

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found.")

    print(f"\n=== Phase {phase} | w{window_size} | {dataset_variant} ===")

    data, labels = load_windowed_data(
        base_dir=processed_dir,
        window_size=window_size,
        variant=dataset_variant,
    )

    y_train_per_phase = create_per_phase_labels(labels['train'], phase)
    y_test_per_phase = create_per_phase_labels(labels['test'], phase)

    # ---- Datasets & Loaders ----
    train_ds = WindowedFlowDataset(data['train'], y_train_per_phase)
    test_ds = WindowedFlowDataset(data['test'], y_test_per_phase)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ---- Model ----
    input_dim = train_ds[0][0].shape[-1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=2,
        with_softmax=False,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    # ---- Class Weights ----
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train_per_phase,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)

    # ---- Training Loop ----
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

    misclassified_info = misclassified_samples(
        y_test_per_phase,
        y_pred,
        labels['test'],
    )

    # ---- Save Artifacts ----
    model_dir = experiment_dir / "models" / f"w{window_size}" / dataset_variant
    results_dir = experiment_dir / "results" / f"w{window_size}" / dataset_variant

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / f"phase_{phase}.pth")

    save_per_phase_metrics(
        acc,
        precision,
        recall,
        f1,
        cm,
        misclassified_info,
        out_file=results_dir / f"phase_{phase}_metrics.json",
    )

    save_loss_plot(
        train_losses,
        epochs,
        out_file=results_dir / f"phase_{phase}_training_loss.png",
    )

    print("Saved model and metrics.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--window_size", type=int, default=10)
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
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/deepproblog/phase_classifiers")

    for phase in range(1, 6):
        train_phase_classifier(
            phase=phase,
            processed_dir=processed_dir,
            experiment_dir=experiment_dir,
            window_size=args.window_size,
            resampled=args.resampled,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
        )
