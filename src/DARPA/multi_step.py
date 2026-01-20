"""
Run a DeepProbLog multi-step experiment using pretrained LSTMs.
"""

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
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from data.dataset import FlowTensorSource, DARPADPLDataset
from network import FlowLSTM


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
    print("Using pretrained models:", pretrained)
    nets = []
    for phase in phases:
        net = FlowLSTM(input_dim, with_softmax=True)

        if pretrained:
            print(f"Loading pretrained model for phase {phase}...")
            model_path = ROOT_DIR / f"pretrained/phase_{phase}.pth"
            net.load_state_dict(torch.load(model_path, map_location="cpu"))

        net_name = f"net{phase}"
        net = Network(
            net,
            net_name, 
            batching=True
        )
        learning_rate = 1e-4 #if pretrained else 1e-3 # assign lr based on pretrained or from scratch
        net.optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        nets.append(net)
    print()

    return nets


def run(function_name, resampled, pretrained, lookback_limit, batch_size=50):
    """ Run a DeepProbLog experiment for the given function name and dataset settings. """
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    pretrained_str = "pretrained" if pretrained else "from_scratch"
    resampled_str = "resampled" if resampled else "original"
    lookback_limit_str = f"lookback{lookback_limit}" if lookback_limit else "full_lookback"
    name = f"{function_name}_{pretrained_str}_{resampled_str}_{lookback_limit_str}"
    print(f"\n=== Running experiment: {name} ===")

    # Prepare datasets
    DARPA_train = FlowTensorSource("train", resampled_str)
    DARPA_test  = FlowTensorSource("test", resampled_str)

    train_set = DARPADPLDataset("train", function_name, resampled_str, lookback_limit, run_id)
    test_set  = DARPADPLDataset("test", function_name, resampled_str, lookback_limit, run_id)
    
    # Load LSTM networks and build DPL model
    print(f"\n--- Initializing networks and building DeepProbLog model ---")
    input_dim = DARPA_train[0][0].shape[-1]
    phases = get_target_phases(function_name)
    nets = load_lstms(input_dim=input_dim, pretrained=pretrained, phases=phases)
    model = Model(
        ROOT_DIR / f"logic/{function_name}.pl",
        nets
    )
    model.set_engine(ExactEngine(model), cache=True)
    model.optimizer = SGD(model, 5e-2)
    model.add_tensor_source("train", DARPA_train)
    model.add_tensor_source("test",  DARPA_test)

    # Train and evaluate
    print(f"\n--- Training {function_name} DeepProbLog model with batch size {batch_size} ---")
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train = train_model(
        model=model, 
        loader=loader, 
        stop_condition=1,   # number of epochs. Maybe increase?
        log_iter=100,
        profile=0,
        # infoloss=0.5,     # regularization term?
    )

    # Save results
    RESULTS_DIR = ROOT_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    snapshot_dir = RESULTS_DIR / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    log_dir = RESULTS_DIR / "logs" / run_id[:8]
    log_dir.mkdir(exist_ok=True)

    model.save_state(f"{snapshot_dir}/" + name + ".pth")
    cm = get_confusion_matrix(model, test_set, verbose=0)
    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment(f"Accuracy {cm.accuracy()}")
    train.logger.comment("Confusion Matrix:\n" + str(cm))
    train.logger.write_to_file(f"{log_dir}/{name}_{run_id[-4:]}")


if __name__ == "__main__":
    # Command: uv run python src/DARPA/multi_step.py --function_name ddos --resampled --pretrained --lookback_limit 20000

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--function_name", type=str, default="multi_step", help="Function name for the dataset")
    ap.add_argument("--resampled", action="store_true", default=False, help="Use resampled dataset")
    ap.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained LSTM models")
    ap.add_argument("--lookback_limit", type=int, default=None, help="Use limited lookback window for dataset preparation.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Set random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run(
        function_name=args.function_name,
        resampled=args.resampled,
        pretrained=args.pretrained,  
        lookback_limit=args.lookback_limit
    )