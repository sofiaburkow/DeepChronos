from collections import Counter
import random
from pathlib import Path
import argparse

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
    sample_data
)

from src.feature_engineering.features import FEATURES


def process_data(
        dataset, 
        scenario, 
        file_name,
        feature_group,
        window_size, 
        sample,
        sample_strategy,
        seed
    ):

    base_interim = Path("data/interim") / dataset / scenario
    base_processed = Path("data/processed") / dataset / scenario / feature_group / "windowed"

    dataset_file = base_interim / "flows_labeled" / file_name
    if not dataset_file.exists():
        raise FileNotFoundError(f"{dataset_file} not found")

    print(f"[+] Loading labeled dataset from {dataset_file}")

    df = pd.read_csv(dataset_file)
    df = sort_by_time(df)
    df["orig_index"] = df.index

    y_phases = df["phase"]

    if feature_group == "all":
        feature_list = FEATURES.all_nn_features
    elif feature_group == "sub":
        feature_list = FEATURES.sub_nn_features

    features_unprocessed, numeric_cols, categorical_cols = filter_features(
        df, 
        feature_list,
    )

    features_processed, pipeline = process_features(
        X=features_unprocessed,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    # convert sparse --> dense once
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

    train_windows, test_windows = temporal_split_windows(windows, 0.6)

    train_data = pack_windows(train_windows)
    test_data  = pack_windows(test_windows)

    phases = set(train_data["y"]) | set(test_data["y"])
    print(f"Phases in dataset: {sorted(phases)}")

    check_phase_coverage(train_data["y"], "Train set", expected_phases=phases)
    check_phase_coverage(test_data["y"], "Test set", expected_phases=phases)

    if sample:
        counts = Counter(train_data["y"])
        print("Before:", counts)

        classes = sorted(set(train_data["y"]))
        classes.remove(0)  # Don't sample benign class
        target = 10000

        if sample_strategy == "up":
            print("\n[+] Upsampling minority classes...")
            train_data = sample_data(
                data=train_data,
                mode="over",
                target_count=target,
                classes=classes,   # ONLY upsample these
                random_state=seed,
            )
        elif sample_strategy == "down":
            print("\n[+] Downsampling majority classes...")
            train_data = sample_data(
                data=train_data,
                mode="under",
                target_count=target,
                classes=[5],   # ONLY downsample these
                random_state=seed,
            )

        print("After:", Counter(train_data["y"]))
    
    # Sanity check
    lengths = {k: len(v) for k, v in train_data.items()} 
    assert len(set(lengths.values())) == 1, f"Mismatch: {lengths}"
    
    # Save data
    config_name = f"w{window_size}/" + (f"{sample_strategy}" if sample else "original")
    output_dir = base_processed / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, values in train_data.items():
        np.save(output_dir / f"{key}_train.npy", values)

    for key, values in test_data.items():
        np.save(output_dir / f"{key}_test.npy", values)

    print(f"[✓] Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Command: uv run python -m src.feature_engineering.windowing --dataset darpa2000 --scenario s2_inside --feature_group dpl --window_size 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--file_name", type=str, default="all_flows_labeled.csv")
    parser.add_argument("--feature_group", type=str, default="sub", choices=["all", "sub"])
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    configs = [
        # {"sample": False, "sample_strategy": None},
        # {"sample": True, "sample_strategy": "up"},
        {"sample": True, "sample_strategy": "down"},
        ]   

    for config in configs:
        sample = config["sample"]
        sample_strategy = config["sample_strategy"]

        print(f"\n=== Processing {args.feature_group} NN features (sample_strategy={sample_strategy}) ===")
        process_data(
            dataset=args.dataset,
            scenario=args.scenario,
            file_name=args.file_name,
            feature_group=args.feature_group,
            window_size=args.window_size,
            sample=sample,
            sample_strategy=sample_strategy,
            seed=args.seed
    )