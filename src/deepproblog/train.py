from pathlib import Path
import argparse
from datetime import datetime
import random
import json

import torch
import numpy as np

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.train import train_model

from src.datasets.flow_datasets import (
    load_windowed_data,
    FlowTensorSource, 
    FlowDPLDataset,
)
from src.networks.flow_lstm import LSTMClassifier
from src.deepproblog.metrics import (
    snapshot_params, 
    print_param_changes, 
    get_confusion_matrix,
    compute_metrics_from_cm, 
    log_metrics,
)


def load_phase_networks(
    input_dim: int,
    phases: list[int],
    pretrained: bool,
    pretrained_dir: Path,
):
    """
    Create FlowLSTM modules and wrap them as DeepProbLog networks.
    """
    wrapped_networks = []
    raw_modules = []
    snapshots_before = []

    for phase in phases:
        net = LSTMClassifier(input_dim=input_dim, with_softmax=True)

        if pretrained:
            if pretrained_dir is None:
                raise ValueError("pretrained_dir must be provided if pretrained=True")

            model_path = pretrained_dir / f"phase_{phase}.pth"
            if not model_path.exists():
                raise FileNotFoundError(model_path)

            print(f"Loading pretrained weights for phase {phase}")
            net.load_state_dict(torch.load(model_path, map_location="cpu"))

        raw_modules.append(net)
        snapshots_before.append(snapshot_params(net))

        wrapped = Network(net, f"net{phase}", batching=True)
        wrapped.optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-4)
        wrapped_networks.append(wrapped)

    return wrapped_networks, raw_modules, snapshots_before


def run_experiment(
    processed_dir: Path,
    experiment_dir: Path,
    logic_file: str,
    window_size: int,
    resampled: bool,
    pretrained: bool,
    batch_size: int,
    verbose: bool,
):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_variant = "resampled" if resampled else "original"
    window_tag = f"w{window_size}"

    experiment_name = (
        f"{logic_file}_"
        f"{'pretrained' if pretrained else 'scratch'}_"
        f"{dataset_variant}_"
        f"{window_tag}"
    )

    cache_id = (
        f"{logic_file}_"
        f"{dataset_variant}_"
        f"{window_tag}"
    )

    print(f"\n=== Running {experiment_name} ===")

    # --- Load Datasets ---

    data, labels, metadata = load_windowed_data(
        base_dir=processed_dir,
        window_size=window_size,
        variant=dataset_variant,
    ) 

    train_tensor_source = FlowTensorSource(data["train"])
    test_tensor_source = FlowTensorSource(data["test"])

    print("Train tensor source size:", len(train_tensor_source))
    print("Test tensor source size:", len(test_tensor_source))

    cache_dir = experiment_dir / f"{logic_file}/cache"
    queries_file_path = experiment_dir / f"{logic_file}/debug_queries" / f"{cache_id}_train_{run_id}.txt"

    train_set = FlowDPLDataset(
        labels=labels["train"],
        metadata=metadata["train"],
        split_name="train",
        logic_file=logic_file,
        cache_dir=cache_dir,
        cache_id=f"{cache_id}_train",
        save_queries=True,
        queries_file=queries_file_path
    )

    test_set = FlowDPLDataset(
        labels=labels["test"],
        metadata=metadata["test"],
        split_name="test",
        logic_file=logic_file,
        cache_dir=cache_dir,
        cache_id=f"{cache_id}_test",
        save_queries=False,
    )

    # --- Build Networks ---

    networks, modules, snapshots_before = load_phase_networks(
        input_dim = train_tensor_source[0].shape[-1],
        phases = [1,2,3,4,5],
        pretrained = pretrained,
        pretrained_dir = \
            experiment_dir / "phase_classifiers/models" / window_tag / dataset_variant,
    )

    # --- Build Model ---

    logic_dir = Path("src/deepproblog/models")
    logic_file_path = logic_dir / f"{logic_file}.pl"
    model = Model(logic_file_path, networks)

    model.set_engine(ExactEngine(model), cache=True)
    model.optimizer = SGD(model, 5e-2)

    model.add_tensor_source("train", train_tensor_source)
    model.add_tensor_source("test", test_tensor_source)

    # --- Train ---

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    print(f"\nTraining with batch_size={batch_size}")
    train = train_model(
        model=model,
        loader=loader,
        stop_condition=1,  # one epoch
        log_iter=100,
        profile=0,
    )

    # --- Debugging ---

    train_set.dump_queries()

    if verbose:
        print("\nParameter changes:")
        print_param_changes(modules, snapshots_before)

    # --- Evaluate ---

    cm, errors = get_confusion_matrix(model, test_set, verbose=1)
    metrics = compute_metrics_from_cm(cm)

    # --- Save Results ---    

    # Save model state
    model_dir = experiment_dir / f"{logic_file}/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{experiment_name}_{run_id}.pth"
    model.save_state(model_path)
    print("Saved model to:", model_path)

    # Save errors
    errors_dir = experiment_dir / f"{logic_file}/errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    errors_path = errors_dir / f"{experiment_name}_{run_id}.json"
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2, default=str)
    print("Saved errors to:", errors_path)

    # Save logs
    logs_dir = experiment_dir / f"{logic_file}/logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{experiment_name}_{run_id}"
    log_metrics(train.logger, metrics, "Dataset results", per_class=True)
    train.logger.comment("Confusion Matrix:\n" + str(cm))
    train.logger.write_to_file(str(log_file))
    print("Saved log to:", log_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--logic_file", type=str, default="darpa_flags")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--resampled", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Define paths
    processed_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/windowed")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/deepproblog")

    run_experiment(
        processed_dir=processed_dir,
        experiment_dir=experiment_dir,
        logic_file=args.logic_file,
        window_size=args.window_size,
        resampled=args.resampled,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )