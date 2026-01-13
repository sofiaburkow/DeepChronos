import json
from collections import Counter
import random

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


def process_data(dataset_file, feature_file, output_dir, resample=True, seed=123):
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
    window_size = 10
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

        output_dir += "resampled/"
    else:
        output_dir += "original/"

    # Save data to disk
    save_data(output_dir, X_train, X_test, y_phases_train, y_phases_test)


if __name__ == "__main__":
    # Command: uv run python src/DARPA/data/process_data.py

    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_file = "src/DARPA/data/raw/flows.csv"
    feature_file = f"src/DARPA/data/features.json"
    output_dir = "src/DARPA/data/processed/"
    resample = False

    process_data(
        dataset_file=dataset_file,
        feature_file=feature_file,
        output_dir=output_dir,
        resample=resample,
        seed=seed
    )

    

    

    