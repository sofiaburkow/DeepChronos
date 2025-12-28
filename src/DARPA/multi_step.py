from json import dumps
from keras.models import load_model
from pathlib import Path

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

from src.DARPA.data.processed import DARPA_train, DARPA_test


method = "exact" # exact inference
N = 1
name = "DARPA_{}_{}".format(method, N)

train_set = DARPA_train
test_set = DARPA_test

# Use pretrained models
nets = []
NUM_PHASES = 5
PRETRAINED_DIR = Path("models/pretrained")
for i in range(1, NUM_PHASES + 1):
    model_path = PRETRAINED_DIR / f"phase_{i}.h5"
    keras_model = load_model(str(model_path))
    net = Network(keras_model, f"phase_{i}_net", batching=True)
    net.optimizer = torch.optim.Adam(net.network_module.parameters(), lr=1e-3)
    nets.append(net)

# Build DPL multi-step attack model
model = Model("multi_step.pl", nets)
if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "geometric_mean":
    model.set_engine(
        ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    )

model.add_tensor_source("train", DARPA_train)
model.add_tensor_source("test", DARPA_test)

loader = DataLoader(train_set, 2, False)
train = train_model(model, loader, 1, log_iter=100, profile=0)
model.save_state("snapshot/" + name + ".pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name)