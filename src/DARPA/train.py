"""
Run a DeepProbLog multi-step experiment using pretrained LSTMs.
"""

from pathlib import Path
import argparse

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from data.dataset import FlowTensorSource, DARPADPLDataset
from network import FlowLSTM


ROOT_DIR = Path(__file__).parent

def load_lstms(input_dim: int, pretrained: bool):
    nets = []
    for phase in range(1, 6):
        net = FlowLSTM(input_dim, with_softmax=True)

        if pretrained:
            print(f"Loading pretrained model for phase {phase}...")
            model_path = ROOT_DIR / f"pretrained/phase_{phase}.pth"
            net.load_state_dict(torch.load(model_path, map_location="cpu"))

        net_name = f"phase_{phase}_net"
        net = Network(
            net,
            net_name, 
            batching=True
        )

        net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        nets.append(net)

    return nets

def run(pretrained):

    DARPA_train = FlowTensorSource("train")
    DARPA_test  = FlowTensorSource("test")

    function_name = "multi_step"
    train_set = DARPADPLDataset("train", function_name)
    test_set  = DARPADPLDataset("test", function_name)
    # Use pretrained models
    input_dim = DARPA_train[0][0].shape[-1]
    print(f"Input dim: {input_dim}")

    nets = load_lstms(input_dim=input_dim, pretrained=pretrained)
    
    # Build DPL multi-step attack model
    model_path = ROOT_DIR / "model.pl"
    model = Model(model_path, nets)

    method = "exact"
    if method == "exact":
        model.set_engine(ExactEngine(model), cache=True)
    else:
        model.set_engine(
            ApproximateEngine(model, 1, ApproximateEngine.geometric_mean)
        )

    model.add_tensor_source("train", DARPA_train)
    model.add_tensor_source("test",  DARPA_test)

    # loader = DataLoader(train_set, batch_size=32, shuffle=True)
    loader = DataLoader(train_set, batch_size=32, shuffle=False)
    train = train_model(
        model=model, 
        loader=loader, 
        stop_condition=1,
    )

    print(get_confusion_matrix(model, test_set).accuracy())


if __name__ == "__main__":
    # Command: uv run python src/DARPA/train.py

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=bool, default=True, help="Use pretrained LSTM models")
    args = ap.parse_args()

    print("Using pretrained models:", args.pretrained)
    
    run(pretrained=args.pretrained)