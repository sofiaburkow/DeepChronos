from pathlib import Path
import numpy as np
import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from data import DARPADataset
from network import FlowLSTM, FlowLSTMWrapper, FlowTensorSource


def load_datasets():
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")

    return X_train, X_test, y_train, y_test


def load_pretrained_lstm(model_path, input_dim):
    lstm = FlowLSTM(input_dim)
    lstm.load_state_dict(torch.load(model_path, map_location="cpu"))
    lstm.eval()

    return lstm


def run(method="exact"):
    X_train, X_test, y_train, y_test = load_datasets()
    train_set = DARPADataset(y_train)
    test_set  = DARPADataset(y_test)


    # Use pretrained models
    nets = []
    for phase in range(1, 6):
        lstm = load_pretrained_lstm(
            Path("models/pretrained") / f"phase_{phase}.pth",
            input_dim=X_train.shape[-1],
        )
        wrapper = FlowLSTMWrapper(lstm)
        net = Network(
            wrapper,
            f"phase_{phase}_net", 
            batching=True
        )
        nets.append(net)

    # Build DPL multi-step attack model
    model = Model("models/multi_step.pl", nets)
    if method == "exact":
        model.set_engine(ExactEngine(model), cache=True)
    else:
        model.set_engine(
            ApproximateEngine(model, 1, ApproximateEngine.geometric_mean)
        )

    model.add_tensor_source("train", FlowTensorSource(X_train))
    model.add_tensor_source("test",  FlowTensorSource(X_test))

    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    train = train_model(model, loader, epochs=1)

    print(get_confusion_matrix(model, test_set).accuracy())


if __name__ == "__main__":
    run()