from collections import Counter
import random
from pathlib import Path
import argparse
from itertools import product

import pandas as pd
import numpy as np
import torch

from src.feature_engineering.utils import (
    sort_by_time,
    filter_features,
    process_features,
    build_sequences,
    pack_windows,
    check_phase_coverage
)

from src.feature_engineering.features import FEATURES


def process_one_net_data(
        dataset, 
        scenario,
        file_name,
        feature_group, 
        window_size,
        pipeline=None
    ):
    # Load dataset
    data_dir = Path("data/interim") / dataset / scenario
    dataset_file = data_dir / "flows_labeled" / f"{file_name}.csv"

    if not dataset_file.exists():
        raise FileNotFoundError(f"{dataset_file} not found")
    
    print(f"[+] Loading labeled dataset from {dataset_file}")

    df = pd.read_csv(dataset_file)
    df = sort_by_time(df)
    df["orig_index"] = df.index

    y_phases = df["phase"]

    # Process features
    if feature_group == "full":
        feature_list = FEATURES.full_nn_features
    elif feature_group == "reduced":
        feature_list = FEATURES.reduced_nn_features
    elif feature_group == "aug":
        feature_list = FEATURES.aug_nn_features
    else:
        raise ValueError(f"Unknown feature group: {feature_group}")

    features_unprocessed, numeric_cols, categorical_cols = filter_features(
        df, 
        feature_list,
    )

    if pipeline:
        features_processed = pipeline.transform(features_unprocessed)
    else:   
        features_processed, pipeline = process_features(
            X=features_unprocessed,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )

    if hasattr(features_processed, "toarray"):
        features_processed = features_processed.toarray()

    windows = build_sequences(
        df,
        X=features_processed,
        y=y_phases.values,
        window_size=window_size,
        feature_spec=FEATURES
    )

    print("NN feature shape:", features_processed.shape)
    print("Window X shape:", windows[0]["X"].shape)

    return windows, pipeline


def process_data(
        dataset, 
        scenario,
        file_name,
        feature_group,
        window_size,
    ):

    scenarios = scenario.split("_")
    if len(scenarios) == 2: # ait
        train_scenario = scenarios[0]
        test_scenario = scenarios[1]
    elif len(scenarios) == 4: # darpa
        train_scenario = scenarios(f"{scenarios[0]}_{scenarios[1]}")
        test_scenario = scenarios(f"{scenarios[2]}_{scenarios[3]}")

    train_windows, pipeline = process_one_net_data(dataset, train_scenario, file_name, feature_group, window_size, None)
    test_windows, _ = process_one_net_data(dataset, test_scenario, file_name, feature_group, window_size, pipeline)
    assert train_windows[0]["X"].shape[1] == test_windows[0]["X"].shape[1], "Feature dimension mismatch between train and test sets"

    train_data = pack_windows(train_windows)
    test_data  = pack_windows(test_windows)

    phases = set(train_data["y"]) | set(test_data["y"])
    print(f"Phases in dataset: {sorted(phases)}")
    check_phase_coverage(train_data["y"], "Train set", expected_phases=phases)
    check_phase_coverage(test_data["y"], "Test set", expected_phases=phases)
    
    # --- Save Processed Data ---
    out_dir = Path("data/processed") / dataset / scenario / feature_group / "windowed" / f"w{window_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, values in train_data.items():
        np.save(out_dir / f"{key}_train.npy", values)
    for key, values in test_data.items():
        np.save(out_dir / f"{key}_test.npy", values)

    print(f"[✓] Saved processed data to {out_dir}")


if __name__ == "__main__":
    # Command: uv run python -m src.feature_engineering.windowing_cross --dataset aitv2 --scenario santos_fox

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside_s1_dmz")
    parser.add_argument("--file_name", type=str, default="flows_augmented")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    feature_groups = [
        "full", 
        "reduced",
        "aug"
    ]
    window_sizes = [
        10,
        100
    ]

    for feature_group, window_size in product(feature_groups, window_sizes):
        print(f"\n=== Processing {feature_group} NN features w{window_size} ===")
        process_data(
            dataset=args.dataset,
            scenario=args.scenario,
            file_name=args.file_name,
            feature_group=feature_group,
            window_size=window_size,
    )