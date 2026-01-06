"""
Feature processing and window-building helpers for DARPA dataset.
"""

import os
import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


def sort_by_time(df):
    '''
    Sort dataframe by start_time.
    '''
    df = df.copy()
    df = df.sort_values("start_time").reset_index(drop=True)
    
    return df


def filter_features(df, feature_list):
    '''
    Filter dataframe on the given feature list.
    Return filtered dataframe, numeric columns, and categorical columns.
    '''
    # Define all possible features
    numeric_cols = [
        "start_time","end_time","duration", 
        "sport","dport","orig_bytes","resp_bytes",
        "missed_bytes","orig_pkts","orig_ip_bytes",
        "resp_pkts","resp_ip_bytes",
    ]
    categorical_cols = [
        "proto","service","conn_state", 
        "local_orig","local_resp","history",    
        "tunnel_parents","ip_proto",   
    ]

    # Filter on features
    df_filtered = df[feature_list]
    numeric_cols = [col for col in numeric_cols if col in feature_list]
    categorical_cols = [col for col in categorical_cols if col in feature_list]
    
    return df_filtered, numeric_cols, categorical_cols


def process_features(X, numeric_cols, categorical_cols):
    '''
    Process features using a pipeline with StandardScaler for numerical
    and OneHotEncoder for categorical features.
    Return processed features along with the pipeline.
    '''
    transformer = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(with_mean=False), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
        ], 
        sparse_threshold = 0.3 # return sparse matrix if sparsity > 30%
    )
    pipeline = Pipeline(steps=[("transform", transformer)])

    X_processed = pipeline.fit_transform(X)

    return X_processed, pipeline


def check_phase_coverage(y_phases, split_name, expected_phases={0,1,2,3,4,5}):
    """
    Print phase counts and fail if any expected phase is missing.
    """
    phases, counts = np.unique(y_phases, return_counts=True)
    phase_counts = dict(zip(phases.tolist(), counts.tolist()))

    print(f"\n{split_name} phase distribution:")
    for p in sorted(expected_phases):
        print(f"  Phase {p}: {phase_counts.get(p, 0)}")

    missing = expected_phases - set(phase_counts.keys())
    if missing:
        raise ValueError(
            f"{split_name} is missing phases: {sorted(missing)}"
        )
    
    print("All phases are present in dataset.")

    return phase_counts


def temporal_split(flow_ids, X, y_phases, train_ratio, window_size, random_state):
    '''
    Temporal split that ensures both train and test contain flows from all phases.
    '''
    flow_ids_shuffled, X_shuffled, y_phases_shuffled = shuffle(flow_ids, X, y_phases, random_state=random_state)
    split_idx = int(len(X_shuffled) * train_ratio) - window_size + 1

    # Split data
    flow_ids_train = flow_ids_shuffled[:split_idx]
    X_train = X_shuffled[:split_idx]
    y_phases_train = y_phases_shuffled[:split_idx]

    flow_ids_test  = flow_ids_shuffled[split_idx:]
    X_test  = X_shuffled[split_idx:]
    y_phases_test = y_phases_shuffled[split_idx:]

    # Validate phase coverage
    check_phase_coverage(y_phases_train, "Train set")
    check_phase_coverage(y_phases_test, "Test set")

    return flow_ids_train, flow_ids_test, X_train, X_test, y_phases_train, y_phases_test


def build_sequences(X, y, flow_ids, window_size):
    X_sequences, y_sequences, flow_id_sequences = [], [], []
    for i in range(X.shape[0] - window_size + 1):
        X_window = X[i:i+window_size]
        y_window = y[i+window_size-1] # predict next-step state
        flow_id_window = flow_ids[i+window_size-1] 
        X_sequences.append(X_window)
        y_sequences.append(y_window)
        flow_id_sequences.append(flow_id_window)

    return np.array(X_sequences), np.array(y_sequences), np.array(flow_id_sequences)


def prepare_phase_dataset(y_phases, target_phase):
    '''
    Prepare labels for a specific target phase:
    - Create binary label where 1 indicates the target phase, else 0
    '''
    y_phases = y_phases.copy()

    # Binary label for target phase
    y_phase = (y_phases == target_phase).astype(int)

    return y_phase


if __name__ == "__main__":
    # Command: uv run python src/DARPA/data/process_data.py

    seed = 123

    dataset_file = "src/DARPA/data/raw/flows.csv"
    output_dir = "src/DARPA/data/processed/"
    feature_file = f"src/DARPA/data/features.json"

    # Load dataset and feature list 
    df = pd.read_csv(dataset_file)
    df = sort_by_time(df)

    with open(feature_file) as f:
        feature_list = json.load(f)

    # Prepare features
    features_unprocessed, numeric_cols, categorical_cols = filter_features(df, feature_list)
    # TODO: not best practice to fit on test data
    features_processed, pipeline = process_features(
        X=features_unprocessed, 
        numeric_cols=numeric_cols, 
        categorical_cols=categorical_cols
    )

    # Build sequences
    flow_ids = df['flow_id']
    y_phases = df['phase']

    window_size = 10
    X_sequences, y_phases_sequences, flow_id_sequences = build_sequences(
        X=features_processed, 
        y=y_phases, 
        flow_ids=flow_ids, 
        window_size=window_size
    )

    # Split data
    flow_ids_train, flow_ids_test, X_train, X_test, y_phases_train, y_phases_test = temporal_split(
        flow_ids=flow_id_sequences,
        X=X_sequences,
        y_phases=y_phases_sequences,
        train_ratio=0.6,
        window_size=10,
        random_state=seed
    )

    # Save data to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving data to {output_dir}...")
    np.save(output_dir / "flow_ids_train.npy", flow_ids_train)
    np.save(output_dir / "flow_ids_test.npy", flow_ids_test)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_phases_train)
    np.save(output_dir / "y_test.npy", y_phases_test)

    # Save binary labels for attack vs benign
    y_train_binary = (y_phases_train >= 1).astype(int)
    y_test_binary  = (y_phases_test >= 1).astype(int)
    np.save(output_dir / "y_train_binary.npy", y_train_binary)
    np.save(output_dir / "y_test_binary.npy", y_test_binary)
    
    # Save labels for each phase
    for phase in range(1,6):
        # Save full data for each phase
        y_train = prepare_phase_dataset(y_phases_train, target_phase=phase)
        y_test = prepare_phase_dataset(y_phases_test, target_phase=phase)
        np.save(output_dir / f"y_phase_{phase}_train.npy", y_train)
        np.save(output_dir / f"y_phase_{phase}_test.npy", y_test)
        
        # Save attack-only test data for each phase 
        # attack_mask_test  = y_test  >= 1
        # X_test_attack = X_test[attack_mask_test]
        # y_test_attack = np.ones(len(X_test_attack), dtype=int)
        # np.save(output_dir / f"X_phase_{phase}_attack_test.npy", X_test_attack)
        # np.save(output_dir / f"y_phase_{phase}_attack_test.npy", y_test_attack)   

    print("Finished saving all data.")

    