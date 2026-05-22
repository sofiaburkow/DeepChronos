from pathlib import Path
import argparse
from datetime import datetime
import random

import torch
from torch.optim import Adam
import numpy as np

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.train import train_model
from deepproblog.utils.stop_condition import EpochStop, StopOnNoChange

from src.datasets.flow_datasets import (
    load_windowed_data,
    FlowTensorSource, 
    FlowDPLDataset,
)
from src.networks.lstm import LSTMClassifier
from src.evaluation.eval import get_confusion_matrix
from src.evaluation.metrics import (
    extract_cm,
    compute_metrics, 
    log_metrics,
    save_dpl_metrics,
)
from src.evaluation.plots import save_plots


def load_networks(
    input_dim: int,
    num_networks: int,
    pretrained: bool,
    pretrained_dir: Path,
    learning_rate: float,
):
    """
    Create FlowLSTM modules and wrap them as DeepProbLog networks.
    """
    wrapped_networks = []
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

        wrapped = Network(net, f"net{phase}", batching=True)
        wrapped.optimizer = Adam(wrapped.parameters(), lr=learning_rate)
        wrapped_networks.append(wrapped)

    return wrapped_networks


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
    learning_rate: float,
    epochs: int,
):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_tag = f"w{window_size}"

    experiment_name = (
        f"{logic_file}_"
        f"{'pretrained' if pretrained else 'endtoend'}_"
        f"{feature_group}_"
        f"{window_tag}_"
        f"{subset}_"
        f"{learning_rate}lr"
    )

    cache_id = (
        f"{logic_file}_"
        f"{window_tag}_"
        f"{subset}"
    )

    print(f"\n=== Running {experiment_name} ===")

    # --- Load Datasets ---
    data, labels, logic_features = load_windowed_data(
        data_dir=data_dir,
        subset=subset,
    ) 

    train_tensor_source = FlowTensorSource(data["train"])
    test_tensor_source = FlowTensorSource(data["test"])

    print("Train tensor source shape:", train_tensor_source[0].shape)
    print("Test tensor source shape:", test_tensor_source[0].shape)

    cache_dir = experiment_dir / f"{logic_file}/cache"

    train_set = FlowDPLDataset(
        labels=labels["train"],
        logic_features=logic_features["train"],
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
        split_name="test",
        logic_file=logic_file,
        cache_dir=cache_dir,
        cache_id=f"{cache_id}_test",
        save_queries=True,
        queries_file= \
            experiment_dir / f"{logic_file}/debug_queries" / f"{cache_id}_test_{run_id}.txt"
    )

    # --- Build Networks ---
    networks = load_networks(
        input_dim = train_tensor_source[0].shape[-1],
        num_networks= num_networks,
        pretrained = pretrained,
        pretrained_dir = pretrained_dir,
        learning_rate=learning_rate,
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
    log_iter = 10 if subset != "full" else 100
    stop_condition = EpochStop(epochs) | StopOnNoChange(attribute="loss", patience=5)
    print(f"\nTraining with batch_size={batch_size}")
    train = train_model(
        model=model,
        loader=loader,
        stop_condition=stop_condition,
        log_iter=log_iter,
        profile=0,
    )
    # train_set.dump_queries()

    # Save model state
    model_dir = experiment_dir / f"{logic_file}/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{experiment_name}_{run_id}.pth"
    model.save_state(model_path)
    print("Saved model to:", model_path)

    # --- Evaluate ---
    if "darpa2000" in data_dir.parts:
        class_order = ["benign", "phase1", "phase2", "phase3", "phase4", "phase5"]
    elif "aitv2" in data_dir.parts:
        class_order = ["benign", "phase1", "phase2", "phase3", "phase4"]

    cm, errors, correct, inference_times, y_true, y_pred = \
        get_confusion_matrix(model, test_set, classes=class_order, verbose=1)
    mat, classes = extract_cm(cm)
    metrics = compute_metrics(y_true, y_pred, mat, classes, layout="pred_actual")

    save_dpl_metrics(
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        cm=mat,
        metrics=metrics,
        classes=classes,
        inference_times=inference_times,
        errors=errors,
        correct=correct,
    )

    log_metrics(
        logger=train.logger, 
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        metrics=metrics,
        per_class=True,
        inference_times=inference_times,
        cm=cm
    )

    save_plots(
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        logger=train.logger,
        cm=mat,
        classes=classes,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--experiment_dir", type=Path)
    parser.add_argument("--pretrained_dir", type=Path)
    parser.add_argument("--logic_file", type=str, default="darpa")
    parser.add_argument("--feature_group", type=str, default="aug")
    parser.add_argument("--num_networks", type=int, default=1)
    parser.add_argument("--subset", type=str, default="full")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_dpl_model(
        data_dir=args.data_dir,
        experiment_dir=args.experiment_dir,
        pretrained_dir=args.pretrained_dir,
        logic_file=args.logic_file,
        feature_group=args.feature_group,
        num_networks=args.num_networks,
        subset=args.subset,
        window_size=args.window_size,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )