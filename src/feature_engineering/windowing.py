import json
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
    check_phase_coverage,
    resample_data,
)


def process_data(dataset, scenario_network, feature_file, window_size, resample, seed):

    base_interim = Path("data/interim") / dataset / scenario_network
    base_processed = Path("data/processed") / dataset / scenario_network / "windowed"

    dataset_file = base_interim / "flows_labeled" / "all_flows_labeled.csv"

    if not dataset_file.exists():
        raise FileNotFoundError(f"{dataset_file} not found")

    print(f"[+] Loading labeled dataset from {dataset_file}")
    df = pd.read_csv(dataset_file)
    df = sort_by_time(df)
    df["orig_index"] = df.index

    with open(feature_file) as f:
        feature_list = json.load(f)

    # Labels
    y_phases = df["phase"]

    # Feature processing
    features_unprocessed, numeric_cols, categorical_cols = \
        filter_features(df, feature_list)

    features_processed, pipeline = process_features(
        X=features_unprocessed,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    # Build windows
    windows = build_sequences(
        df=df,
        X=features_processed,
        y=y_phases,
        window_size=window_size
    )

    # Temporal split
    train_windows, test_windows = temporal_split_windows(
        windows=windows,
        train_ratio=0.6
    )

    train_data = {
        "X": np.array([w["X"] for w in train_windows]),
        "y": np.array([w["phase"] for w in train_windows]),
        "t": np.array([w["t"] for w in train_windows]),
        "src_ip": np.array([w["src_ip"] for w in train_windows]),
        "dst_ip": np.array([w["dst_ip"] for w in train_windows]),
        "start_time": np.array([w["start_time"] for w in train_windows]),
        "orig_index": np.array([w["orig_index"] for w in train_windows]),
    }

    test_data = {
        "X": np.array([w["X"] for w in test_windows]),
        "y": np.array([w["phase"] for w in test_windows]),
        "t": np.array([w["t"] for w in test_windows]),
        "src_ip": np.array([w["src_ip"] for w in test_windows]),
        "dst_ip": np.array([w["dst_ip"] for w in test_windows]),
        "start_time": np.array([w["start_time"] for w in test_windows]),
        "orig_index": np.array([w["orig_index"] for w in test_windows]),
    }

    check_phase_coverage(train_data["y"], "Train set")
    check_phase_coverage(test_data["y"], "Test set")

    if resample:
        print("\n[+] Upsampling minority classes...")

        counts = Counter(train_data["y"])
        print("Before:", counts)

        train_data = resample_data(
            data=train_data,
            target_count=counts.get(5, max(counts.values())),
            phases=[1, 2, 3, 4, 5],
            random_state=seed
        )

        print("After:", Counter(train_data["y"]))
    
    # Sanity check
    lengths = {k: len(v) for k, v in train_data.items()} 
    assert len(set(lengths.values())) == 1, f"Mismatch: {lengths}"
    
    # Save data
    config_name = f"w{window_size}/" + ("resampled" if resample else "original")
    output_dir = base_processed / config_name

    for key, values in train_data.items():
        np.save(output_dir / f"{key}_train.npy", values)

    for key, values in test_data.items():
        np.save(output_dir / f"{key}_test.npy", values)

    print(f"[✓] Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Command: uv run python -m src.feature_engineering.windowing --window_size 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario_network", type=str, default="s1_inside")
    parser.add_argument("--feature_file", type=str, default="src/feature_engineering/feature_list.json")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    for resample in [False, True]:
        print(f"\n=== Processing (resample={resample}) ===")
        process_data(
            dataset=args.dataset,
            scenario_network=args.scenario_network,
            feature_file=args.feature_file,
            window_size=args.window_size,
            resample=resample,
            seed=args.seed
        )
