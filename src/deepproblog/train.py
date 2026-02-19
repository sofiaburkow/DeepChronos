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
from deepproblog.evaluate import get_confusion_matrix

from src.datasets.flow_datasets import (
    load_windowed_data,
    FlowTensorSource, 
    FlowDPLDataset,
)
from src.networks.flow_lstm import LSTMClassifier
from src.deepproblog.metrics import (
    snapshot_params, 
    print_param_changes, 
    get_filtered_dataset, 
    compute_metrics_from_cm, 
    log_metrics,
)

    
def get_target_phases(function_name: str) -> list[int]:
    if function_name == "ddos":
        return [5]
    if function_name == "multi_step":
        return [1, 2, 3, 4, 5]
    raise ValueError(f"Unknown function_name: {function_name}")


def load_phase_networks(
    input_dim: int,
    phases: list[int],
    pretrained: bool,
    pretrained_dir: Path | None,
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
    function_name: str,
    processed_dir: Path,
    logic_dir: Path,
    experiment_dir: Path,
    window_size: int,
    resampled: bool,
    pretrained: bool,
    lookback_limit: int | None,
    batch_size: int,
    debug: bool = False,
):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = "resampled" if resampled else "original"
    window_tag = f"w{window_size}"

    experiment_name = (
        f"{function_name}_"
        f"{'pretrained' if pretrained else 'scratch'}_"
        f"{variant}_"
        f"{'lookback' + str(lookback_limit) if lookback_limit else 'full'}_"
        f"{window_tag}"
    )

    cache_id = (
        f"{function_name}_"
        f"{variant}_"
        f"{'lookback' + str(lookback_limit) if lookback_limit else 'full'}_"
        f"{window_tag}"
    )

    print(f"\n=== Running {experiment_name} ===")

    # --- Load Datasets ---

    data, labels = load_windowed_data(
        base_dir=processed_dir,
        window_size=window_size,
        variant=variant,
    ) 

    train_tensor_source = FlowTensorSource(data["train"])
    test_tensor_source = FlowTensorSource(data["test"])

    print("Train tensor source size:", len(train_tensor_source))
    print("Test tensor source size:", len(test_tensor_source))


    train_set = FlowDPLDataset(
        labels=labels["train"],
        split_name="train",
        function_name=function_name,
        lookback_limit=lookback_limit,
        cache_dir=experiment_dir / f"{function_name}/cache",
        cache_id=f"{cache_id}_train",
        save_queries=True,
        queries_file=experiment_dir / f"{function_name}/debug_queries/{cache_id}_train_{run_id}.txt"
    )

    test_set = FlowDPLDataset(
        labels=labels["test"],
        split_name="test",
        function_name=function_name,
        lookback_limit=lookback_limit,
        cache_dir=experiment_dir / f"{function_name}/cache",
        cache_id=f"{cache_id}_test",
        save_queries=True,
        queries_file=experiment_dir / f"{function_name}/debug_queries/{cache_id}_test_{run_id}.txt"
    )

    # --- Build Networks ---

    input_dim = train_tensor_source[0].shape[-1]
    phases = get_target_phases(function_name)

    pretrained_dir = (
        experiment_dir / "phase_classifiers/models" / window_tag / variant
        if pretrained else None
    )

    networks, modules, snapshots_before = load_phase_networks(
        input_dim=input_dim,
        phases=phases,
        pretrained=pretrained,
        pretrained_dir=pretrained_dir,
    )

    # --- Build DeepProbLog Model ---

    logic_file = logic_dir / f"{function_name}.pl"
    model = Model(logic_file, networks)

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
        stop_condition=1,  # one epoch (adjust later)
        log_iter=100,
        profile=0,
    )

    # For debugging purposes: write queries to file
    train_set.dump_queries()
    test_set.dump_queries()

    if debug:
        print("\nParameter changes:")
        print_param_changes(modules, snapshots_before)

    # --- Evaluate ---

    cm = get_confusion_matrix(model, test_set, verbose=0)
    metrics = compute_metrics_from_cm(cm)

    # --- Save Results ---

    model_dir = experiment_dir / f"{args.function_name}/models"
    logs_dir = experiment_dir / f"{args.function_name}/logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{experiment_name}_{run_id}.pth"
    model.save_state(model_path)

    if metrics:
        log_metrics(train.logger, metrics, "Full dataset results", per_class=True)

    train.logger.comment("Confusion Matrix:\n" + str(cm))

    if function_name == "ddos":
        filter_name = "all_prev_phases"
        filtered_test_set = get_filtered_dataset(test_set, filter_name)
        if filtered_test_set is None:
            train.logger.comment("No filtered test examples found.")
        else:
            cm_filtered = get_confusion_matrix(
                model, filtered_test_set, verbose=0
            )
            metrics_f = compute_metrics_from_cm(cm_filtered)
            if metrics_f is not None:
                log_metrics(
                    train.logger,
                    metrics_f,
                    title=f"Filtered test set metrics ({filter_name.replace('_', ' ')})",
                    per_class=True,
                )
            train.logger.comment(
                "\nConfusion Matrix:\n" + str(cm_filtered)
            )

    log_file = logs_dir / f"{experiment_name}_{run_id}"
    train.logger.write_to_file(str(log_file))

    print("Saved model to:", model_path)
    print("Saved log to:", log_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--function_name", default="multi_step")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--resampled", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lookback_limit", type=int)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    processed_dir = Path(f"data/processed/{args.dataset}/{args.scenario}/windowed")
    experiment_dir = Path(f"experiments/{args.dataset}/{args.scenario}/deepproblog")
    logic_dir = Path("src/deepproblog/logic")

    run_experiment(
        function_name=args.function_name,
        processed_dir=processed_dir,
        logic_dir=logic_dir,
        experiment_dir=experiment_dir,
        window_size=args.window_size,
        resampled=args.resampled,
        pretrained=args.pretrained,
        lookback_limit=args.lookback_limit,
        batch_size=args.batch_size,
    )