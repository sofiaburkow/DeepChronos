"""
Helper functions for processing DARPA DPL dataset.
"""

from pathlib import Path
from collections import Counter

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


def sort_by_time(df):
    """
    Sort dataframe by start_time.
    """
    df = df.copy()
    df = df.sort_values("start_time").reset_index(drop=True)
    
    return df


def filter_features(df, feature_list):
    """
    Filter dataframe on the given feature list.
    Return filtered dataframe, numeric columns, and categorical columns.
    """
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
    """
    Process features using a pipeline with StandardScaler for numerical
    and OneHotEncoder for categorical features.
    Return processed features along with the pipeline.
    """
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


def build_sequences(X, y, window_size):
    windows = []
    for i in range(X.shape[0] - window_size + 1):
        windows.append({
            "t": i,                     # original time index
            "X": X[i:i+window_size],    # window of features
            "phase": y[i+window_size-1] # label at the end of the window
        })

    return windows


def temporal_split_windows(windows, train_ratio):
    indices = np.arange(len(windows))

    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=[w["phase"] for w in windows],
        shuffle=True,
        random_state=42
    )

    train_windows = sorted(
        [windows[i] for i in train_idx],
        key=lambda w: w["t"]
    )

    test_windows = sorted(
        [windows[i] for i in test_idx],
        key=lambda w: w["t"]
    )

    return train_windows, test_windows


def resample_indices(y, sampling_strategy, random_state=123):
    """
    Resample indices based on the given sampling strategy using RandomOverSampler.
    Return resampled indices and corresponding labels.
    """
    indices = np.arange(len(y)).reshape(-1, 1)

    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    resampled_indices, y_resampled = ros.fit_resample(indices, y)
    return resampled_indices.flatten(), y_resampled


def resample_data(X, y, desired_target, phases, random_state=123):
    """
    Upsample minority classes in y to reach the desired target count.
    Return resampled X and y.
    """
    counts = Counter(y)
    sampling_strategy = {p: desired_target for p in phases if counts.get(p, 0) < desired_target}
        
    idx_resampled, y_resampled = resample_indices(
        y,
        sampling_strategy=sampling_strategy
    )

    X_resampled = [X[i] for i in idx_resampled]

    return X_resampled, y_resampled


def prepare_phase_dataset(y_phases, target_phase):
    """
    Prepare labels for a specific target phase:
    - Create binary label where 1 indicates the target phase, else 0
    """
    y_phases = y_phases.copy()

    # Binary label for target phase
    y_phase = (y_phases == target_phase).astype(int)

    return y_phase


def save_data(output_dir, X_train, X_test, y_train, y_test):
    """
    Save training and testing data along with their labels to the specified output directory.
    """
    print(f"\nSaving data to {output_dir}...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)

    # Save multi-class labels
    np.save(output_dir / "y_train_multi_class.npy", y_train)
    np.save(output_dir / "y_test_multi_class.npy", y_test)

    print("Finished saving all data.")