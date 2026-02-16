import json
from collections import Counter
import random
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch

from helper_func import (
    sort_by_time,
    filter_features,
    process_features,
    build_sequences,
    temporal_split_windows,
    check_phase_coverage,
    resample_data,
    save_data
)


def process_data(dataset_file, feature_file, output_dir, window_size, resample, seed):
    """
    Process DARPA DPL dataset: load, preprocess, build sequences, split, oversample, and save.
    """
    # Load dataset and feature list 
    print(f"Loading dataset from {dataset_file} and features from {feature_file}...")
    df = pd.read_csv(dataset_file)
    df = sort_by_time(df)
    with open(feature_file) as f:
        feature_list = json.load(f)

    # Extract labels
    y_phases = df['phase']

    # Prepare features
    features_unprocessed, numeric_cols, categorical_cols = filter_features(df, feature_list)
    
    # TODO: not best practice to fit on test data
    features_processed, pipeline = process_features(
        X=features_unprocessed, 
        numeric_cols=numeric_cols, 
        categorical_cols=categorical_cols
    )

    # Build sequences
    train_ratio = 0.6
    windows = build_sequences(
        X=features_processed, 
        y=y_phases,
        window_size=window_size
    )

    # Split data
    train_windows, test_windows = temporal_split_windows(
        windows=windows,
        train_ratio=train_ratio
    )
    X_train = np.array([w["X"] for w in train_windows])
    y_phases_train = np.array([w["phase"] for w in train_windows])
    X_test = np.array([w["X"] for w in test_windows])
    y_phases_test = np.array([w["phase"] for w in test_windows])

     # Validate phase coverage
    check_phase_coverage(y_phases_train, "Train set")
    check_phase_coverage(y_phases_test, "Test set")

    if resample:
        # Oversample minority classes in training set
        print("\nUpsampling minority classes in training set...")
        counts = Counter(y_phases_train)
        print("Distribution before upsampling:", counts)
        X_train_resampled, y_train_resampled = resample_data(
            X=X_train, 
            y=y_phases_train,
            desired_target=counts[5],
            phases=[1,2,3,4,5],
            random_state=seed
        )
        counts_upsampled = Counter(y_train_resampled)
        print("Distribution after upsampling:", counts_upsampled)

        X_train = X_train_resampled
        y_phases_train = y_train_resampled

    # Save data to disk
    config = f"w{window_size}/" +(f"resampled" if resample else "original")
    output_dir = Path(output_dir) / config
    save_data(output_dir, X_train, X_test, y_phases_train, y_phases_test)


if __name__ == "__main__":
    # Command: uv run python src/DARPA/data/process_data.py

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_file", default="src/DARPA/data/raw/flows.csv")
    ap.add_argument("--feature_file", default="src/DARPA/data/features.json")
    ap.add_argument("--output_dir", default="src/DARPA/data/processed", help="Output directory to save the processed dataset")
    ap.add_argument("--window_size", type=int, default=10, help="Size of the time window for the features")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Process both original and resampled datasets
    for resample in [False, True]:
        print(f"\n=== Processing dataset (resample={resample}) ===")
        process_data(
            dataset_file=args.dataset_file,
            feature_file=args.feature_file,
            output_dir=args.output_dir,
            window_size=args.window_size,
            resample=resample,
            seed=seed
        )

    

    

    