from pathlib import Path
import argparse
import numpy as np
from sklearn import multiclass
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
from src.evaluation.lstm_metrics import (
    eval, 
    misclassified_samples,
    save_per_phase_metrics,
)
from src.evaluation.dpl_metrics import compute_metrics
from src.evaluation.plots import save_loss_plot


def create_binary_labels(y_multi_class: np.ndarray) -> np.ndarray:
    """
    Convert multi-class labels into binary labels (benign/attack). 
    """
    return (y_multi_class != 0).astype(np.int64)


def create_per_phase_labels(y_multi_class: np.ndarray, phase: int) -> np.ndarray:
    """
    Convert multi-class labels into binary labels for a specific phase.
    """
    return (y_multi_class == phase).astype(np.int64)


def train_classifier(
    per_phase: bool,
    phase: int,
    processed_dir: Path,
    experiment_dir: Path,
    window_size: int,
    dataset_variant: str,
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",
):

    dataset_path = processed_dir / f"w{window_size}" / dataset_variant

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found.")

    data, labels, _, _ = load_windowed_data(
        base_dir=processed_dir,
        window_size=window_size,
        dataset_variant=dataset_variant,
    )
    if per_phase:
        file_name = f"phase_{phase}"
        print(f"\n=== Phase {phase} | w{window_size} | {dataset_variant} ===")
        y_train = create_per_phase_labels(labels['train'], phase)
        y_test = create_per_phase_labels(labels['test'], phase)
    else:
        file_name = "multiclass"
        print(f"\n=== Multiclass | w{window_size} | {dataset_variant} ===")
        y_train = create_binary_labels(labels['train'])
        y_test = create_binary_labels(labels['test'])
    
    # ---- Datasets & Loaders ----
    train_ds = WindowedFlowDataset(data['train'], y_train)
    test_ds = WindowedFlowDataset(data['test'], y_test)

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
        y=y_train,
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
    cm, classes, y_pred = eval(model, test_loader, multi_class=False, device=device)
    metrics = compute_metrics(cm, classes, layout="actual_pred")

    misclassified_info = misclassified_samples(
        y_test,
        y_pred,
        labels['test'],
    )

    # ---- Save Artifacts ----
    model_dir = experiment_dir / "models" / f"w{window_size}" / dataset_variant
    results_dir = experiment_dir / "results" / f"w{window_size}" / dataset_variant

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / f"{file_name}.pth")

    save_per_phase_metrics(
        cm=cm,
        metrics=metrics,
        mis_info=misclassified_info,
        out_file=results_dir / f"{file_name}_metrics.json",
    )

    save_loss_plot(
        train_losses,
        epochs,
        out_file=results_dir / f"{file_name}_training_loss.png",
    )

    print("Saved model and metrics.")


if __name__ == "__main__":
    # uv run python -m src.deepproblog.pretrain --dataset darpa2000 --scenario s1_inside --window_size 10 --dataset_variant up

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--dataset_variant", type=str, default="original")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processed_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/dpl/windowed")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/pretrained_nets")

    # Train a separate classifier for each phase (1-5)
    for phase in range(1, 6): # phases 1-5
        train_classifier(
            per_phase=True,
            phase=phase,
            processed_dir=processed_dir,
            experiment_dir=experiment_dir,
            window_size=args.window_size,
            dataset_variant=args.dataset_variant,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
        )

    # Train multiclass classifier
    train_classifier(
        per_phase=False,
        phase=None,
        processed_dir=processed_dir,
        experiment_dir=experiment_dir,
        window_size=args.window_size,
        dataset_variant=args.dataset_variant,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )
