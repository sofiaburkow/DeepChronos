from collections import Counter

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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


def build_sequences(df, X, y, window_size, feature_spec):
    windows = []

    logic_cols = feature_spec.logic_features
    meta_cols = feature_spec.metadata_features

    for i in range(X.shape[0] - window_size + 1):

        row_idx = i + window_size - 1

        window = {
            "X": X[i:i+window_size],
            "y": y[row_idx],
            "t": i,
        }

        # logic features
        for col in logic_cols:
            window[col] = df[col].iloc[row_idx]

        # metadata
        for col in meta_cols:
            window[col] = df[col].iloc[row_idx]

        windows.append(window)

    return windows


def temporal_split_windows(windows, train_ratio):
    indices = np.arange(len(windows))

    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=[w["y"] for w in windows],
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


def pack_windows(windows):
    keys = windows[0].keys()

    return {
        k: np.array([w[k] for w in windows])
        for k in keys
    }


def sample_data(
    data,
    mode,
    target_count,
    classes,
    random_state=123,
):
    """
    Resample dataset using index-based sampling.

    Parameters
    ----------
    data : dict
        Dict containing arrays with equal length (must include key "y").
    mode : str
        "over" or "under".
    target_count : int
        Desired number of samples per selected class.
    classes : list
        Classes to resample.
        - oversampling: minority classes
        - undersampling: majority classes
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Resampled dataset.
    """

    y = data["y"]
    counts = Counter(y)

    # --------------------------------------------------
    # Build sampling strategy
    # --------------------------------------------------
    if mode == "over":
        sampler_cls = RandomOverSampler
        sampling_strategy = {
            c: target_count
            for c in classes
            if counts.get(c, 0) < target_count
        }

    elif mode == "under":
        sampler_cls = RandomUnderSampler
        sampling_strategy = {
            c: target_count
            for c in classes
            if counts.get(c, 0) > target_count
        }

    else:
        raise ValueError("mode must be 'over' or 'under'")

    if not sampling_strategy:
        return data

    # --------------------------------------------------
    # Resample indices ONLY (important for sequences)
    # --------------------------------------------------
    indices = np.arange(len(y)).reshape(-1, 1)

    sampler = sampler_cls(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    idx_resampled, y_resampled = sampler.fit_resample(indices, y)
    idx_resampled = idx_resampled.flatten()

    # --------------------------------------------------
    # Apply indices to ALL tensors
    # --------------------------------------------------
    resampled = {
        key: np.asarray(values)[idx_resampled]
        for key, values in data.items()
    }

    resampled["y"] = y_resampled

    return resampled