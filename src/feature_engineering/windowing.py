import json
from collections import Counter
import random
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch

from training.utils import (
    sort_by_time,
    filter_features,
    process_features,
    build_sequences,
    temporal_split_windows,
    check_phase_coverage,
    resample_data,
    save_data
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
    X_test = np.array([w["X"] for w in test_windows])
    y_test = np.array([w["phase"] for w in test_windows])

    check_phase_coverage(y_train, "Train set")
    check_phase_coverage(y_test, "Test set")

    # Optional resampling
    if resample:
        print("\n[+] Upsampling minority classes...")
        counts = Counter(y_train)
        print("Before:", counts)

        X_train, y_train = resample_data(
            X=X_train,
            y=y_train,
            desired_target=counts.get(5, max(counts.values())),
            phases=[1, 2, 3, 4, 5],
            random_state=seed
        )

        print("After:", Counter(y_train))

    # Save output
    config_name = f"w{window_size}/" + ("resampled" if resample else "original")
    output_dir = base_processed / config_name

    save_data(output_dir, X_train, X_test, y_train, y_test)

    print(f"[✓] Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Command: uv run python src/data/windowing.py --dataset darpa2000 --scenario_network s1_inside --feature_file src/data/feature_list.json --window_size 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario_network", required=True)
    parser.add_argument("--feature_file", required=True)
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
