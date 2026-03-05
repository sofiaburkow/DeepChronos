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

    X_train = np.array([w["X"] for w in train_windows])
    y_train = np.array([w["phase"] for w in train_windows])
    t_train = np.array([w["t"] for w in train_windows])
    src_ip_train = np.array([w["src_ip"] for w in train_windows])
    dst_ip_train = np.array([w["dst_ip"] for w in train_windows])
    start_time_train = np.array([w["start_time"] for w in train_windows])

    X_test = np.array([w["X"] for w in test_windows])
    y_test = np.array([w["phase"] for w in test_windows])
    t_test = np.array([w["t"] for w in test_windows])
    src_ip_test = np.array([w["src_ip"] for w in test_windows])
    dst_ip_test = np.array([w["dst_ip"] for w in test_windows])
    start_time_test = np.array([w["start_time"] for w in test_windows])

    check_phase_coverage(y_train, "Train set")
    check_phase_coverage(y_test, "Test set")

    # Optional resampling
    if resample:
        print("\n[+] Upsampling minority classes...")
        counts = Counter(y_train)
        print("Before:", counts)

        X_train, y_train, t_train, src_ip_train, dst_ip_train, start_time_train = resample_data(
            X=X_train,
            y=y_train,
            t=t_train,
            src_ip=src_ip_train,
            dst_ip=dst_ip_train,
            start_time=start_time_train,
            desired_target=counts.get(5, max(counts.values())),
            phases=[1, 2, 3, 4, 5],
            random_state=seed
        )

        print("After:", Counter(y_train))
    
    # Sanity check: all arrays must have the same length
    assert len(X_train) == len(y_train) == len(t_train) == len(src_ip_train) == len(dst_ip_train) == len(start_time_train), \
        "Train set length mismatch"
    assert len(X_test) == len(y_test) == len(t_test) == len(src_ip_test) == len(dst_ip_test) == len(start_time_test), \
        "Test set length mismatch"

    # Save output
    config_name = f"w{window_size}/" + ("resampled" if resample else "original")
    output_dir = base_processed / config_name

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)

    # Save multi-class labels
    np.save(output_dir / "y_train_multi_class.npy", y_train)
    np.save(output_dir / "y_test_multi_class.npy", y_test)

    # Save time indices
    np.save(output_dir / "t_train.npy", t_train)
    np.save(output_dir / "t_test.npy", t_test)

    # Save metadata
    np.save(output_dir / "src_ip_train.npy", src_ip_train)
    np.save(output_dir / "dst_ip_train.npy", dst_ip_train)
    np.save(output_dir / "start_time_train.npy", start_time_train)

    np.save(output_dir / "src_ip_test.npy", src_ip_test)
    np.save(output_dir / "dst_ip_test.npy", dst_ip_test)
    np.save(output_dir / "start_time_test.npy", start_time_test)

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
