import json
from collections import Counter

import pandas as pd

from helper_func import (
    sort_by_time,
    filter_features,
    process_features,
    build_sequences,
    temporal_split,
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
    X_sequences, y_phases_sequences = build_sequences(
        X=features_processed, 
        y=y_phases,
        window_size=window_size
    )

    # Split data
    X_train, X_test, y_phases_train, y_phases_test = temporal_split(
        X=X_sequences,
        y_phases=y_phases_sequences,
        train_ratio=0.6,
        window_size=10,
        random_state=seed
    )

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

    # Save data to disk
    save_data(output_dir, X_train, X_test, y_phases_train, y_phases_test)

if __name__ == "__main__":
    # Command: uv run python src/DARPA/data/process_data.py

    seed = 123

    dataset_file = "src/DARPA/data/raw/flows.csv"
    feature_file = f"src/DARPA/data/features.json"
    output_dir = "src/DARPA/data/processed/"
    resample = True

    process_data(
        dataset_file=dataset_file,
        feature_file=feature_file,
        output_dir=output_dir,
        resample=resample,
        seed=seed
    )

    

    

    