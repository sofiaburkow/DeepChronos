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
    resample_data,
)

from src.feature_engineering.features import FEATURES


def process_data(
        dataset, 
        scenario_network, 
        window_size, 
        resample, 
        seed
    ):

    base_interim = Path("data/interim") / dataset / scenario_network
    base_processed = Path("data/processed") / dataset / scenario_network / "windowed"

    dataset_file = base_interim / "flows_labeled" / "all_flows_labeled.csv"
    if not dataset_file.exists():
        raise FileNotFoundError(f"{dataset_file} not found")

    print(f"[+] Loading labeled dataset from {dataset_file}")

    df = pd.read_csv(dataset_file)
    df = sort_by_time(df)
    df["orig_index"] = df.index

    y_phases = df["phase"]

    features_unprocessed, numeric_cols, categorical_cols = filter_features(
        df, 
        FEATURES.nn_features
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
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, values in train_data.items():
        np.save(output_dir / f"{key}_train.npy", values)

    for key, values in test_data.items():
        np.save(output_dir / f"{key}_test.npy", values)

    print(f"[✓] Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Command: uv run python -m src.feature_engineering.windowing --window_size 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario_network", type=str, default="s1_inside")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    for resample in [False, True]:
        print(f"\n=== Processing (resample={resample}) ===")
        process_data(
            dataset=args.dataset,
            scenario_network=args.scenario_network,
            window_size=args.window_size,
            resample=resample,
            seed=args.seed
        )
