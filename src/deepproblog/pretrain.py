from pathlib import Path
import argparse
import numpy as np
from sklearn import multiclass
import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

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
    data_dir: Path,
    experiment_dir: Path,
    subset: str,
    window_size: int,
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",
):

    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found.")

    data, labels, _, _ = load_windowed_data(
        data_dir=data_dir,
        subset=subset,
    ) 

    if per_phase:
        file_name = f"phase_{phase}"
        print(f"\n=== Phase {phase} ===")
        y_train = create_per_phase_labels(labels['train'], phase)
        y_test = create_per_phase_labels(labels['test'], phase)
    else:
        file_name = "multiclass"
        y_train = create_binary_labels(labels['train'])
        y_test = create_binary_labels(labels['test'])
    
    # ---- Datasets & Loaders ----
    train_dataset = WindowedFlowDataset(data['train'], y_train)
    test_dataset = WindowedFlowDataset(data['test'], y_test)

    sampler = build_weighted_sampler(train_dataset.y)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size
    )

    # ---- Model ----
    input_dim = train_dataset[0][0].shape[-1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=2,
        with_softmax=False,
    ).to(device)

    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

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
    cm, classes, y_pred = eval(model, test_loader, multiclass=False, device=device)
    metrics = compute_metrics(cm, classes, layout="actual_pred")

    misclassified_info = misclassified_samples(
        y_test,
        y_pred,
        labels['test'],
    )

    # ---- Save Artifacts ----
    model_dir = experiment_dir / "models" / f"w{window_size}" / f"{subset}"
    results_dir = experiment_dir / "results" / f"w{window_size}" / f"{subset}"
    
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
        out_file=results_dir/f"{file_name}_training_loss.png",
    )

    plot_confusion_matrix(
        cm=np.array(cm).T, # Transpose to get actual vs predicted layout
        classes=classes,
        out_path=results_dir/f"{file_name}_cm.png",
    )

    print("Saved model and metrics.")


if __name__ == "__main__":
    # uv run python -m src.deepproblog.pretrain --dataset aitv2 --scenario fox

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--feature_group", type=str, default="behavioral")
    parser.add_argument("--subset", type=str, default="full")
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
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/pretrained_nets")

    # if args.dataset == "aitv2":
    #     num_phases = 4
    # elif args.dataset == "darpa2000":
    #     num_phases = 5
    # else:
    #     raise ValueError(f"Unsupported dataset: {args.dataset}")

    # # Train a separate classifier for each phase
    # print(f"\n=== Per-Phase | {args.dataset} | {args.scenario} | {args.feature_group} | w{args.window_size} ===")
    # for phase in range(1, num_phases + 1): 
    #     train_classifier(
    #         per_phase=True,
    #         phase=phase,
    #         data_dir=data_dir,
    #         experiment_dir=experiment_dir,
    #         subset=args.subset,
    #         window_size=args.window_size,
    #         batch_size=args.batch_size,
    #         epochs=args.epochs,
    #         device=device,
    #     )

    # Train multiclass classifier
    print(f"\n=== Multiclass | {args.dataset} | {args.scenario} | {args.feature_group} | w{args.window_size} ===")
    train_classifier(
        per_phase=False,
        phase=None,
        data_dir=data_dir,
        experiment_dir=experiment_dir,
        subset=args.subset,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )
