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

from data.dataset import FlowTensorSource, DARPAMultiStepDataset
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

        net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        nets.append(net)

    return nets


def run(pretrained, function_name, batch_size=50):

    run_id = datetime.now().strftime("%Y%m%d_%H%M")

    # Prepare datasets
    DARPA_train = FlowTensorSource("train")
    DARPA_test  = FlowTensorSource("test")

    train_set = DARPAMultiStepDataset("train", function_name, run_id)
    test_set  = DARPAMultiStepDataset("test", function_name, run_id)
    
    # Load LSTM networks and build DPL model
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
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # OK to be True here
    train = train_model(
        model=model, 
        loader=loader, 
        stop_condition=1,   # number of epochs. Maybe increase?
        log_iter=100,
        profile=0,
        # infoloss=0.5,     # regularization term?
    )

    if pretrained:
        name = f"{function_name}_pretrained_{run_id}"
    else:
        name = f"{function_name}_from_scratch_{run_id}"

    snapshot_dir = ROOT_DIR / "snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    model.save_state(f"{snapshot_dir}/" + name + ".pth")
    
    cm = get_confusion_matrix(model, test_set, verbose=0)

    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment(f"Accuracy {cm.accuracy()}")
    train.logger.comment("Confusion Matrix:\n" + str(cm))

    log_dir = ROOT_DIR / "log"
    log_dir.mkdir(exist_ok=True)
    train.logger.write_to_file(f"{log_dir}/{name}")


if __name__ == "__main__":
    # Command: uv run python src/DARPA/multi_step.py --pretrained --function_name ddos --seed 123

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained LSTM models")
    ap.add_argument("--function_name", type=str, default="ddos", help="Function name for the dataset")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run(
        pretrained=args.pretrained, 
        function_name=args.function_name
    )