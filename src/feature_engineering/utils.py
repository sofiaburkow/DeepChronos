import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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

    numeric_cols = [
        "sport",
        "dport",
        "duration", 
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        "connections_per_src_60s",
        "unique_targets_60s",
        "unique_dports_60s",
        "syn_failure_ratio_60s",
        "reject_ratio_60s",
        "reset_ratio_60s",
        "connections_per_dst_60s",
        "unique_sources_per_dst_60s",
    ]

    categorical_cols = [
        "proto",
        "service",
        "conn_state", 
        "local_orig",
        "local_resp",
        "ip_proto",
    ]

    # Filter on features
    df_filtered = df[feature_list]
    numeric_cols = [col for col in numeric_cols if col in feature_list]
    categorical_cols = [col for col in categorical_cols if col in feature_list]
    
    return df_filtered, numeric_cols, categorical_cols


def process_features(X, numeric_cols, categorical_cols):
    """
    Process features.
    Use MinMaxScaler for numerical features and OneHotEncoder for categorical features.
    Return processed features along with the pipeline.
    """
    transformer = ColumnTransformer(
        transformers=[
            ("numerical", MinMaxScaler(), numeric_cols),
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

    missing = set(expected_phases) - set(phase_counts.keys())
    if missing:
        raise ValueError(
            f"{split_name} is missing phases: {sorted(missing)}"
        )
    
    print("All phases are present in dataset.")

    return phase_counts


def build_sequences(df, X, y, window_size, dpl_features):
    windows = []

    for i in range(X.shape[0] - window_size + 1):

        row_idx = i + window_size - 1

        window = {
            "X": X[i:i+window_size],
            "y": y[row_idx],
            "t": i,
        }

        for col in dpl_features:
            window[col] = df[col].iloc[row_idx]

        windows.append(window)

    return windows


def temporal_split_windows(windows, train_ratio, seed):
    indices = np.arange(len(windows))

    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=[w["y"] for w in windows],
        shuffle=True,
        random_state=seed
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