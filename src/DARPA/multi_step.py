from pathlib import Path
import argparse
from json import dumps
from datetime import datetime
import random

import torch
import numpy as np

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.utils.stop_condition import StopOnPlateau
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from data.dataset import FlowTensorSource, DARPADPLDataset
from network import FlowLSTM
from helper_func import (
    snapshot_params, print_param_changes, create_results_dirs, 
    get_filtered_dataset, compute_metrics_from_cm, log_metrics
)


# Root directory is "src/DARPA"
ROOT_DIR = Path(__file__).parent


def get_target_phases(function_name: str):
    if function_name == "recon":
        return [1, 2]
    elif function_name == "ddos":
        return [5]
    elif function_name == "multi_step":
        return [1, 2, 3, 4, 5]
    else:
        raise ValueError(f"Unknown function name: {function_name}")


def load_lstms(input_dim: int, pretrained: bool, phases: list[int]):
    """Load FlowLSTM instances, optionally load pretrained weights, and
    return (deepproblog Network wrappers, raw pytorch modules, snapshots_before).
    """
    print("Using pretrained models:", pretrained)
    nets = []
    pytorch_modules = []
    snapshots_before = []

    for phase in phases:
        net = FlowLSTM(input_dim, with_softmax=True)

        if pretrained:
            print(f"Loading pretrained model for phase {phase}...")
            model_path = ROOT_DIR / f"pretrained/phase_{phase}.pth"
            net.load_state_dict(torch.load(model_path, map_location="cpu"))

        net_name = f"net{phase}"

        # Keep raw PyTorch module for debugging / snapshots
        pytorch_modules.append(net)
        snapshots_before.append(snapshot_params(net))

        # Wrap into DeepProbLog Network and assign optimizer
        wrapped = Network(net, net_name, batching=True)
        wrapped.optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-4)
        nets.append(wrapped)

    return nets, pytorch_modules, snapshots_before


def run(function_name, resampled, pretrained, lookback_limit, debug=False, batch_size=50):
    """Run the full experiment: prepare data, build model, train, evaluate."""

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    pretrained_str = "pretrained" if pretrained else "from_scratch"
    resampled_str = "resampled" if resampled else "original"
    lookback_limit_str = f"lookback{lookback_limit}" if lookback_limit else "full_lookback"
    name = f"{function_name}_{pretrained_str}_{resampled_str}_{lookback_limit_str}"

    print(f"\n=== Running experiment: {name} ===")

    # Prepare datasets
    DARPA_train = FlowTensorSource("train", resampled_str)
    DARPA_test = FlowTensorSource("test", resampled_str)
    train_set = DARPADPLDataset("train", function_name, resampled_str, lookback_limit, run_id)
    test_set = DARPADPLDataset("test", function_name, resampled_str, lookback_limit, run_id)

    # Load networks
    print(f"\n--- Initializing networks and building DeepProbLog model ---")
    input_dim = DARPA_train[0][0].shape[-1]
    phases = get_target_phases(function_name)
    nets, modules, snapshots_before = load_lstms(input_dim=input_dim, pretrained=pretrained, phases=phases)

    # Build model
    model = Model(ROOT_DIR / f"logic/{function_name}.pl", nets)
    model.set_engine(ExactEngine(model), cache=True)
    model.optimizer = SGD(model, 5e-2)
    model.add_tensor_source("train", DARPA_train)
    model.add_tensor_source("test", DARPA_test)

    # Train
    print(f"\n--- Training {function_name} DeepProbLog model with batch size {batch_size} ---")
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # stop = StopOnPlateau("accuracy", delta=0.005, patience=3, warm_up=5) # patiance of 3 epochs
    stop = 1 # training will run for 1 epoch only

    train = train_model(
        model=model,
        loader=loader,
        stop_condition=stop,
        log_iter=100,
        profile=0,
        # infoloss=0.5,     # regularization term?
    )

    if debug:
        # Simple gradient check 
        print("\n--- Parameter changes after training ---")
        print_param_changes(modules, snapshots_before)

    # Compute metrics and save results
    snapshot_dir, log_dir = create_results_dirs(ROOT_DIR, run_id)
    model.save_state(f"{snapshot_dir}/" + name + ".pth")
    train.logger.comment(dumps(model.get_hyperparameters()))

    # Full confusion matrix
    cm = get_confusion_matrix(model, test_set, verbose=0)
    metrics = compute_metrics_from_cm(cm)
    if metrics is not None:
        log_metrics(
            train.logger,
            metrics,
            title="Results for full dataset",
            per_class=True,   # or False if logs get too verbose
        )
    train.logger.comment("Confusion Matrix:\n" + str(cm))

    # Filtered confusion matrix (optional)
    # filter = True
    # if filter:
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
    
    train.logger.write_to_file(f"{log_dir}/{name}_{run_id[-4:]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--function_name", type=str, default="multi_step", help="Function name for the dataset")
    ap.add_argument("--resampled", action="store_true", default=False, help="Use resampled dataset")
    ap.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained LSTM models")
    ap.add_argument("--lookback_limit", type=int, default=None, help="Use limited lookback window for dataset preparation.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run(
        function_name=args.function_name,
        resampled=args.resampled,
        pretrained=args.pretrained,
        lookback_limit=args.lookback_limit,
    )