"""
Run a DeepProbLog multi-step experiment using pretrained LSTMs.
"""

from pathlib import Path
import argparse
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from data.dataset import FlowTensorSource, DARPADPLDataset
from network import FlowLSTM


# Root directory is "src/DARPA"
ROOT_DIR = Path(__file__).parent

def load_lstms(input_dim: int, pretrained: bool):
    # phases = [1, 2, 3, 4, 5]
    phases = [1, 2]
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


def run(pretrained, function_name, batch_size):

    # Prepare datasets
    DARPA_train = FlowTensorSource("train")
    DARPA_test  = FlowTensorSource("test")

    train_set = DARPADPLDataset("train", function_name)
    test_set  = DARPADPLDataset("test", function_name)
    
    # Load LSTM networks and build DPL model
    input_dim = DARPA_train[0][0].shape[-1]
    nets = load_lstms(input_dim=input_dim, pretrained=pretrained)
    model = Model(
        # ROOT_DIR / "model.pl", 
        ROOT_DIR / "test_logic.pl",
        nets
    )
    model.set_engine(ExactEngine(model), cache=True)
    model.optimizer = SGD(model, 5e-2)
    model.add_tensor_source("train", DARPA_train)
    model.add_tensor_source("test",  DARPA_test)

    # Train and evaluate
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # TODO: shuffle=True?
    train = train_model(
        model=model, 
        loader=loader, 
        stop_condition=1, # number of epochs
        log_iter=100,
        profile=0,
        infoloss=0.75,     # regularization term 
    )
    name = f"dpl_multi_step_{function_name}"
    model.save_state("snapshot/" + name + ".pth")
    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment(
        "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
    )
    train.logger.write_to_file("log/" + name)


if __name__ == "__main__":
    # Command: uv run python src/DARPA/multi_step.py

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true", default=False, help="Use pretrained LSTM models")
    ap.add_argument("--function_name", type=str, default="multi_step", help="Function name to use in the model")
    ap.add_argument("--batch_size", type=int, default=50, help="Batch size for training")
    args = ap.parse_args()

    print("Using pretrained models:", args.pretrained)
    print("Using function name:", args.function_name)
    
    run(pretrained=args.pretrained, function_name=args.function_name)