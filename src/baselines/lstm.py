from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import StratifiedKFold

from src.networks.lstm import EnsembleLSTMClassifier, LSTMClassifier
from src.datasets.flow_datasets import load_windowed_data, WindowedFlowDataset
from src.evaluation.eval import eval_lstm
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics, save_metrics_json
from src.evaluation.plots import plot_train_val_loss, plot_train_loss, plot_confusion_matrix


def make_dir(out_dir, subpath):
    path = Path(out_dir) / subpath
    path.mkdir(parents=True, exist_ok=True)

    return path


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


def build_model(classifier, input_dim, num_classes, num_attack_classes, device):
    if classifier == "ensemble":
        model = EnsembleLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=64,
            n_networks=num_attack_classes,
            output_dim=num_classes,
            with_softmax=False,
        ).to(device)
    else:
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=num_classes,
            with_softmax=False,
        ).to(device)

    return model


def train_one_fold(
    model, train_loader, val_loader,
    criterion, optimizer, epochs,
    early_stopping=True, patience=5, min_delta=0.0, device="cpu"
):
    best_loss = np.inf
    best_state, train_losses, val_losses = None, [], []
    epochs_no_improve = 0
    metrics_val = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    val_loss += criterion(model(Xv), yv).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

            if early_stopping:
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    if val_loader:
        cm, classes, y_true, y_pred = eval_lstm(model, val_loader, multiclass=True, device=device)
        metrics_val = compute_metrics(y_true, y_pred, cm, classes, layout="actual_pred")

    return model, train_losses, val_losses, metrics_val


def train_lstm(
    classifier: str,
    data_dir: Path,
    out_dir: Path,
    subset: str,
    feature_group: str,
    window_size: int,
    learning_rate: float,
    batch_size: int, 
    epochs: int = 50,
    cv_folds: int = 5,
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 0.0,
    device: str = "cpu",
    seed: int = 123,
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (f"{classifier}_{feature_group}features_w{window_size}_{subset}data_{learning_rate}lr_{run_id}")
    print(f"\n=== Running {experiment_name} ===")

    # --- Load Data ---
    data, labels, _ = load_windowed_data(
        data_dir=data_dir,
        subset=subset,
    ) 
    train_dataset_full = WindowedFlowDataset(data['train'], labels['train'])
    test_dataset  = WindowedFlowDataset(data['test'], labels['test'])
    y_train_full = np.asarray(labels['train'])
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    
    num_classes = len(set(labels['train']))
    num_attack_classes = num_classes - 1
    input_dim = train_dataset_full[0][0].shape[-1]
    criterion = CrossEntropyLoss()

    # ---- Training ----
    if cv_folds == 1:
        folds = [(np.arange(len(y_train_full)), np.array([], dtype=int))]
    else:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(len(y_train_full)), y_train_full))

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold_idx}/{len(folds)} ---")
        
        if len(val_idx) == 0: 
            train_subset = train_dataset_full
            val_subset = None
        else:
            train_subset = Subset(train_dataset_full, train_idx)
            val_subset = Subset(train_dataset_full, val_idx)

        sampler = build_weighted_sampler(y_train_full[train_idx])
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_subset, batch_size=batch_size) if val_subset else None

        model = build_model(classifier, input_dim, num_classes, num_attack_classes, device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # ---- Train this fold ----
        model, tr_losses, val_losses, metrics_val = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            device=device,
        )

        # ---- Save fold artifacts ----
        if cv_folds > 1 and metrics_val:
            fold_dir = make_dir(out_dir, f"fold_{fold_idx}")
            torch.save(model.state_dict(), fold_dir / f"{experiment_name}_fold{fold_idx}.pth")
            plot_train_val_loss(tr_losses, val_losses or None, fold_dir / f"{experiment_name}_fold{fold_idx}_loss.png")
            fold_metrics.append(metrics_val)

    if cv_folds > 1 and fold_metrics:
        metrics_dir = make_dir(out_dir, "metrics")
        metrics = aggregate_fold_metrics(fold_metrics)
        save_metrics_json(
            cm=None, 
            metrics=metrics,
            out_file=metrics_dir / f"{experiment_name}.json",
        )
        print("\n=== Cross-validation Summary ===")
        for i, fm in enumerate(fold_metrics, start=1):
            print(f"Fold {i}: acc={fm['accuracy']:.4f}, micro_f1={fm['micro_f1']:.4f}, macro_f1={fm['macro_f1']:.4f}")
        print(f"Mean macro-F1: {metrics['macro_f1']:.4f}")
        
    else: # if no CV
        # Evaluate on test set
        cm, classes, y_true, y_pred = eval_lstm(model, test_loader, multiclass=True, device=device)
        metrics = compute_metrics(y_true, y_pred, cm, classes, layout="actual_pred")
        
        # ---- Save Results ----
        model_dir = make_dir(out_dir, "models")
        pred_dir = make_dir(out_dir, "predictions")
        metrics_dir = make_dir(out_dir, "metrics")
        cm_dir = make_dir(out_dir, "cm_plots")
        loss_plot_dir = make_dir(out_dir, "loss_plots")

        torch.save(
            model.state_dict(), 
            model_dir / f"{experiment_name}.pth"
        )

        t_test = np.load(data_dir / "t_test.npy")
        np.savez_compressed(
            pred_dir / f"{experiment_name}.npz",
            flow_idx=t_test + window_size - 1,
            y_true=y_true,
            y_pred=y_pred,
        )
        print(f"Saved predictions to: {pred_dir / f'{experiment_name}.npz'}")

        save_metrics_json(cm, metrics, out_file=metrics_dir / f"{experiment_name}.json")
        
        plot_train_loss(tr_losses, epochs, loss_plot_dir / f"{experiment_name}.png")

        plot_confusion_matrix(
            cm=np.array(cm).T, # Transpose to get actual vs predicted layout
            classes=classes,
            out_path = cm_dir / f"{experiment_name}.png",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default="multiclass")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--subset", type=str, default="full")
    parser.add_argument("--feature_group", type=str, default="base")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cv_folds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_lstm(
        classifier=args.classifier, 
        data_dir=args.data_dir, 
        out_dir=args.out_dir,
        subset=args.subset,
        feature_group=args.feature_group,  
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size, 
        epochs=args.epochs,
        cv_folds=args.cv_folds,
        device=device,
        seed=args.seed,
    )