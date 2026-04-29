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
    temporal_split_windows,
    pack_windows,
    check_phase_coverage,
)

from src.feature_engineering.features import FEATURES


def process_data(
        data_path, 
        out_dir,
        feature_group,
        window_size,
        split_ratio,
        seed
    ):

    print(f"[+] Loading labeled dataset from {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    df = pd.read_csv(data_path)
    df = sort_by_time(df)
    df["orig_index"] = df.index

    y_phases = df["phase"]

    if feature_group == "full":
        feature_list = FEATURES.full_nn_features
    elif feature_group == "reduced":
        feature_list = FEATURES.reduced_nn_features
    elif feature_group == "aug":
        feature_list = FEATURES.aug_nn_features
    else:
        raise ValueError(f"Unknown feature group: {feature_group}")

    # --- Process Features ---
    features_unprocessed, numeric_cols, categorical_cols = filter_features(
        df, 
        feature_list,
    )

    features_processed, pipeline = process_features(
        X=features_unprocessed,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    windows = build_sequences(
        df,
        X=features_processed,
        y=y_phases.values,
        window_size=window_size,
        feature_spec=FEATURES
    )

    # --- Split Data ---
    print("NN feature shape:", features_processed.shape)
    print("Window X shape:", windows[0]["X"].shape)

    train_windows, test_windows = temporal_split_windows(windows, split_ratio, seed=seed)
    train_data = pack_windows(train_windows)
    test_data  = pack_windows(test_windows)

    phases = set(train_data["y"]) | set(test_data["y"])
    print(f"Phases in dataset: {sorted(phases)}")
    check_phase_coverage(train_data["y"], "Train set", expected_phases=phases)
    check_phase_coverage(test_data["y"], "Test set", expected_phases=phases)
    
    # --- Save Processed Data ---
    output_dir = out_dir / f"w{window_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, values in train_data.items():
        np.save(output_dir / f"{key}_train.npy", values)
    for key, values in test_data.items():
        np.save(output_dir / f"{key}_test.npy", values)

    print(f"[✓] Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Command: uv run python -m src.feature_engineering.windowing --dataset aitv2 --scenario santos

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--file_name", type=str, default="flows_augmented.csv")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_path = Path("data/interim") / args.dataset / args.scenario / "flows_labeled" / args.file_name

    feature_groups = [
        "full", 
        "reduced",
        "aug"
    ]
    window_sizes = [
        10,
        100
    ]

    split_ratio = 0.6

    for feature_group, window_size in product(feature_groups, window_sizes):
        print(f"\n=== Processing {feature_group} NN features w{window_size} ===")
        out_dir = Path("data/processed") / args.dataset / args.scenario / feature_group / "windowed"
        process_data(
            data_path=data_path,
            out_dir=out_dir,
            feature_group=feature_group,
            window_size=window_size,
            split_ratio=split_ratio,
            seed=args.seed
        )