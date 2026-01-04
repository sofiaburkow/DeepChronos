from pathlib import Path
import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from data.dataset import DARPAWindowed, DARPAOperator
from network import FlowLSTM, FlowLSTMWrapper

ROOT_DIR = Path(__file__).parent

def load_pretrained_lstm(model_path, input_dim):
    net = FlowLSTM(input_dim)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    # net.eval()
    return net


def run(method="exact"):

    DARPA_train = DARPAWindowed("train")
    DARPA_test  = DARPAWindowed("test")
    input_dim = DARPA_train[0][0].shape[-1]
    print(f"Input dim: {input_dim}")

    function_name = "multi_step"
    train_set = DARPAOperator("train", function_name)
    test_set  = DARPAOperator("test", function_name)

    # Use pretrained models
    nets = []
    for phase in range(1, 6):
    # for phase in range(1, 2):
        pretrained_lstm = load_pretrained_lstm(
            ROOT_DIR / f"models/pretrained/phase_{phase}.pth",
            input_dim=input_dim,
        )

        wrapper = FlowLSTMWrapper(pretrained_lstm)

        net_name = f"phase_{phase}_net"
        net = Network(
            wrapper,
            net_name, 
            batching=True
        )
        nets.append(net)

    # Build DPL multi-step attack model
    model_path = ROOT_DIR / "models/multi_step.pl"
    model = Model(model_path, nets)
    if method == "exact":
        model.set_engine(ExactEngine(model), cache=True)
    else:
        model.set_engine(
            ApproximateEngine(model, 1, ApproximateEngine.geometric_mean)
        )

    model.add_tensor_source("train", DARPA_train)
    model.add_tensor_source("test",  DARPA_test)

    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    train = train_model(
        model=model, 
        loader=loader, 
        stop_condition=1,
    )

    print(get_confusion_matrix(model, test_set).accuracy())


if __name__ == "__main__":
    run()