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
from src.evaluation.dpl_metrics import (
    snapshot_params, 
    print_param_changes, 
    get_confusion_matrix,
    extract_cm,
    compute_metrics, 
    log_metrics,
    save_metrics,
)
from src.evaluation.plots import (
    plot_train_loss,
    plot_confusion_matrix,
)


def load_networks(
    input_dim: int,
    num_networks: int,
    pretrained: bool,
    pretrained_dir: Path,
):
    """
    Create FlowLSTM modules and wrap them as DeepProbLog networks.
    """
    wrapped_networks = []
    raw_modules = []
    snapshots_before = []

    for _, phase in enumerate(range(1, num_networks+1)):
        net = LSTMClassifier(input_dim=input_dim, with_softmax=True)

        if pretrained:
            if pretrained_dir is None:
                raise ValueError("pretrained_dir must be provided if pretrained=True")

            if num_networks == 1:
                print(f"Loading pretrained weights for multiclass model")
                model_path = pretrained_dir / "multiclass.pth"
            else:
                print(f"Loading pretrained weights for phase {phase}")
                model_path = pretrained_dir / f"phase_{phase}.pth"

            if not model_path.exists():
                raise FileNotFoundError(model_path)

            net.load_state_dict(torch.load(model_path, map_location="cpu"))

        raw_modules.append(net)
        snapshots_before.append(snapshot_params(net))

        wrapped = Network(net, f"net{phase}", batching=True)
        wrapped.optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-4)
        wrapped_networks.append(wrapped)

    return wrapped_networks, raw_modules, snapshots_before


def train_dpl_model(
    processed_dir: Path,
    experiment_dir: Path,
    pretrained_dir: Path,
    logic_file: str,
    num_networks: int,
    window_size: int,
    dataset_variant: str,
    pretrained: bool,
    batch_size: int,
    verbose: bool,
):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    data, labels, logic_features, metadata_features = load_windowed_data(
        base_dir=processed_dir,
        window_size=window_size,
        dataset_variant=dataset_variant,
    ) 

    train_tensor_source = FlowTensorSource(data["train"])
    test_tensor_source = FlowTensorSource(data["test"])

    print("Train tensor source size:", len(train_tensor_source))
    print("Test tensor source size:", len(test_tensor_source))

    cache_dir = experiment_dir / f"{logic_file}/cache"

    train_set = FlowDPLDataset(
        labels=labels["train"],
        logic_features=logic_features["train"],
        metadata_features=metadata_features["train"],
        split_name="train",
        logic_file=logic_file,
        cache_dir=cache_dir,
        cache_id=f"{cache_id}_train",
        save_queries=False,  # Set to True to save queries for debugging
        # queries_file= \
        #     experiment_dir / f"{logic_file}/debug_queries" / f"{cache_id}_train_{run_id}.txt"
    )

    test_set = FlowDPLDataset(
        labels=labels["test"],
        logic_features=logic_features["test"],
        metadata_features=metadata_features["test"],
        split_name="test",
        logic_file=logic_file,
        cache_dir=cache_dir,
        cache_id=f"{cache_id}_test",
        save_queries=False,
    )

    # --- Build Networks ---

    networks, modules, snapshots_before = load_networks(
        input_dim = train_tensor_source[0].shape[-1],
        num_networks= num_networks,
        pretrained = pretrained,
        pretrained_dir = pretrained_dir/window_tag/dataset_variant,
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

    # train_set.dump_queries()

    if verbose:
        print("\nParameter changes:")
        print_param_changes(modules, snapshots_before)
    
    # --- Evaluate ---

    cm, errors = get_confusion_matrix(model, test_set, verbose=1)
    mat, classes = extract_cm(cm)
    metrics = compute_metrics(mat, classes, layout="pred_actual")

    # --- Save results ---

    metrics_dir = experiment_dir / f"{logic_file}/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(
        cm = cm, 
        metrics = metrics,
        out_path = metrics_dir / f"{experiment_name}_{run_id}.npz", 
    )

    plot_dir = experiment_dir / f"{logic_file}/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_train_loss(
        logger=train.logger,
        out_path = plot_dir / f"{experiment_name}_{run_id}_loss.png",
    )

    plot_confusion_matrix(
        cm=mat,
        classes=classes,
        experiment_name=experiment_name,
        out_path = plot_dir / f"{experiment_name}_{run_id}_cm.png",
    )

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
    log_metrics(train.logger, experiment_name, metrics, per_class=True)
    train.logger.comment("\nConfusion Matrix:\n" + str(cm))
    train.logger.write_to_file(str(log_file))
    print("Saved log to:", log_file)


if __name__ == "__main__":
    # uv run python -m src.deepproblog.train --scenario s1_dmz --logic_file darpa_neg

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--feature_group", type=str, default="sub", choices=["all", "sub"])
    parser.add_argument("--logic_file", type=str, default="darpa_flags")
    parser.add_argument("--num_networks", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--dataset_variant", type=str, default="original")
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
    processed_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/{args.feature_group}/windowed")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/deepproblog")
    
    scenario_parts = args.scenario.split("_")
    pretrained_tag = f"{scenario_parts[0]}_{scenario_parts[1]}"
    pretrained_dir = Path(f"experiments/{args.dataset}/{pretrained_tag}/pretrained_nets/models")

    train_dpl_model(
        processed_dir=processed_dir,
        experiment_dir=experiment_dir,
        pretrained_dir=pretrained_dir,
        logic_file=args.logic_file,
        num_networks=args.num_networks,
        window_size=args.window_size,
        dataset_variant=args.dataset_variant,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )