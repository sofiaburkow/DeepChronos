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
        net = LSTMClassifier(input_dim=input_dim, output_dim=2, with_softmax=True)

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
    data_dir: Path,
    experiment_dir: Path,
    pretrained_dir: Path,
    logic_file: str,
    feature_group: str,
    num_networks: int,
    subset: str,
    window_size: int,
    pretrained: bool,
    batch_size: int,
    verbose: bool,
):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_tag = f"w{window_size}"

    experiment_name = (
        f"{logic_file}_"
        f"{'pretrained' if pretrained else 'scratch'}_"
        f"{feature_group}_"
        f"{window_tag}_"
        f"{subset}"
    )

    cache_id = (
        f"{logic_file}_"
        f"{window_tag}_"
        f"{subset}"
    )

    print(f"\n=== Running {experiment_name} ===")

    # --- Load Datasets ---

    data, labels, logic_features, metadata_features = load_windowed_data(
        data_dir=data_dir,
        subset=subset,
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
        save_queries=True,  # Set to True to save queries for debugging
        queries_file= \
            experiment_dir / f"{logic_file}/debug_queries" / f"{cache_id}_train_{run_id}.txt"
    )

    test_set = FlowDPLDataset(
        labels=labels["test"],
        logic_features=logic_features["test"],
        metadata_features=metadata_features["test"],
        split_name="test",
        logic_file=logic_file,
        cache_dir=cache_dir,
        cache_id=f"{cache_id}_test",
        save_queries=True,
        queries_file= \
            experiment_dir / f"{logic_file}/debug_queries" / f"{cache_id}_test_{run_id}.txt"
    )

    # --- Build Networks ---
    networks, modules, snapshots_before = load_networks(
        input_dim = train_tensor_source[0].shape[-1],
        num_networks= num_networks,
        pretrained = pretrained,
        pretrained_dir = pretrained_dir,
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

    if subset != "full":
        log_iter = 10
    else:
        log_iter = 100

    print(f"\nTraining with batch_size={batch_size}")
    train = train_model(
        model=model,
        loader=loader,
        stop_condition=50,  # number of epochs
        log_iter=log_iter,
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
    print("Saved log to:", f"{log_file}.log")


if __name__ == "__main__":
    # uv run python -m src.deepproblog.train --dataset aitv2 --scenario fox --logic_file ait_neg_alt --num_networks 4 --feature_group all --fraction 10 

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--feature_group", type=str, default="behavioral", choices=["full", "reduced", "behavioral"])
    parser.add_argument("--logic_file", type=str, default="darpa_neg")
    parser.add_argument("--num_networks", type=int, default=5)
    parser.add_argument("--subset", type=str, default="500b20a")
    parser.add_argument("--window_size", type=int, default=10)
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
    data_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/{args.feature_group}/windowed/w{args.window_size}")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/deepproblog")

    scenario_parts = args.scenario.split("_")
    if len(scenario_parts) == 1:
        pretrained_tag = scenario_parts[0]
    else:
        pretrained_tag = f"{scenario_parts[0]}_{scenario_parts[1]}"
    pretrained_dir = Path(f"experiments/{args.dataset}/{pretrained_tag}/pretrained_nets/models/w{args.window_size}")

    train_dpl_model(
        data_dir=data_dir,
        experiment_dir=experiment_dir,
        pretrained_dir=pretrained_dir,
        logic_file=args.logic_file,
        feature_group=args.feature_group,
        num_networks=args.num_networks,
        subset=args.subset,
        window_size=args.window_size,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )