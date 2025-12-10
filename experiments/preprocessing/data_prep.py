import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump
from scipy.sparse import save_npz


def encode_ip_none(df):
    """Drop IP address fields completely."""
    df = df.copy()
    return df.drop(columns=["src_ip", "dst_ip"]), None


def encode_ip_integer(df):
    """Convert IPv4 x.x.x.x to an integer: a*256^3 + b*256^2 + c*256 + d."""
    def ip_to_int(ip):
        try:
            a, b, c, d = map(int, ip.split("."))
            return (a << 24) + (b << 16) + (c << 8) + d
        except:
            return 0
    df = df.copy()
    df["src_ip_int"] = df["src_ip"].apply(ip_to_int)
    df["dst_ip_int"] = df["dst_ip"].apply(ip_to_int)

    return df.drop(columns=["src_ip", "dst_ip"]), ["src_ip_int", "dst_ip_int"]


IP_ENCODERS = {
    "none": encode_ip_none,
    "integer": encode_ip_integer,
}


def prepare_data(df,feature_set, ip_encoding="none"):
    '''
    Prepare data by filtering on selected features.

    Args:
        df: Input DataFrame with flow data.
        feature_set: List of features to retain.
        ip_encoding: IP address encoding method ("none", "integer", "onehot").
    Returns:
        df: Prepared DataFrame with selected features.
        numeric_cols: List of numeric feature column names.
        categorical_cols: List of categorical feature column names.
        ip_feature_cols: List of IP feature column names.
    '''
    # Drop metadata fields
    df = df.drop(columns=["flow_id", "attack_id", "phase", "attack"])

    # Handle IP address fields
    if ip_encoding not in IP_ENCODERS:
        raise ValueError(f"Unknown IP encoding: {ip_encoding}")

    df, ip_feature_cols = IP_ENCODERS[ip_encoding](df)

    # Categorical features
    categorical_cols = [
        "proto",      # categorical: "tcp", "udp", "icmp", etc.
        "service",    # categorical: "http", "ftp", "dns", etc.
        "conn_state", # categorical: "S0", "S1", "SF", etc.
        "local_orig", # binary flags
        "local_resp", # binary flags
        ]
    
    # Numerical features
    numeric_cols = [
        "start_time",
        "end_time",
        "duration", 
        "sport",
        "dport",
        "orig_bytes", 
        "resp_bytes",
        "orig_pkts", 
        "resp_pkts",
    ]

    # Filter on features
    categorical_cols = [col for col in categorical_cols if col in feature_set]
    numeric_cols = [col for col in numeric_cols if col in feature_set]

    ip_feature_cols = ip_feature_cols if ip_feature_cols else []
    numeric_cols.extend(ip_feature_cols)

    df = df[feature_set + ip_feature_cols]

    return df, numeric_cols, categorical_cols, ip_feature_cols


def construct_pipeline(numeric_cols, categorical_cols):
    '''
    Construct a preprocessing pipeline for numerical and categorical features.
    Args:
        numeric_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.
    Returns:
        pipeline: sklearn Pipeline object for preprocessing. 
    '''
    transformer = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(with_mean=False), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
        ], 
        sparse_threshold = 1.0 # always return sparse matrix
    )
    pipeline = Pipeline(steps=[("transform", transformer)])

    return pipeline


def save_processed_data(output_dir, X_train, y_train, X_test, y_test, pipeline, numeric_cols, categorical_cols, ip_encoding):
    '''
    Save processed data and preprocessing pipeline to disk.
    Args:
        output_dir: Directory to save processed data.
        X_train: Processed training feature matrix.
        y_train: Training labels.
        X_test: Processed testing feature matrix.
        y_test: Testing labels.
        pipeline: Preprocessing pipeline.
        numeric_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.
        ip_encoding: IP encoding method used.
    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    save_npz(os.path.join(output_dir, "X_train.npz"), X_train)
    save_npz(os.path.join(output_dir, "X_test.npz"), X_test)

    dump(pipeline, os.path.join(output_dir, "feature_pipeline.joblib"))

    with open(os.path.join(output_dir, "feature_info.txt"), "w") as f:
        f.write("Numerical features:\n")
        f.write(str(numeric_cols) + "\n\n")
        f.write("Categorical features:\n")
        f.write(str(categorical_cols) + "\n\n")
        f.write(f"IP encoding: {ip_encoding}\n")

    print(f"Saved X, y, and preprocessing pipeline to {output_dir}/")


def prep_process_save_data(df_train, df_test, feature_set, ip_encoding, output_dir, save=True, label_name="attack"):
    '''
    This function takes sampled training data, processes it along with test data,
    and saves the processed datasets and pipeline to the specified output directory.
    '''
    # Prepare features
    df_train_features, numeric_cols, categorical_cols, _ = prepare_data(
        df_train, feature_set, ip_encoding=ip_encoding,     
    )
    df_test_features, _, _, _ = prepare_data(
        df_test, feature_set, ip_encoding=ip_encoding,
    )

    # Process data with pipeline
    pipeline = construct_pipeline(numeric_cols, categorical_cols)
    X_train = pipeline.fit_transform(df_train_features)
    y_train = df_train[label_name]
    X_test = pipeline.transform(df_test_features)
    y_test = df_test[label_name]

    # Save processed data to disk
    if save:
        save_processed_data(output_dir, X_train, y_train, X_test, y_test, pipeline, numeric_cols, categorical_cols, ip_encoding)


if __name__ == "__main__":
    # Command: uv run python experiments/preprocessing/prepare_data.ipynb

    # Configuration
    SEED = 123
    ip_encoding = "none"
    feature_set = ["duration", "proto", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

    # Load dataset
    SCENARIO_ONE_INSIDE_CSV = "data/DARPA_2000/Scenario_One/inside/inside_labeled_flows_all.csv"
    df = pd.read_csv(SCENARIO_ONE_INSIDE_CSV)
    df.head()

    # Split Data into Train/Test
    train_size = 0.6
    test_size = 1 - train_size
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df["attack"], random_state=SEED 
    )
    print("Train set shape:", df_train.shape)
    print("Test set shape:", df_test.shape)

    # Prepare features
    df_train_features, numeric_cols, categorical_cols, ip_feature_cols = prepare_data(
        df_train, feature_set, ip_encoding=ip_encoding,     
    )
    df_test_features, _, _, _ = prepare_data(
        df_test, feature_set, ip_encoding=ip_encoding,
    )
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)
    print("IP feature cols:", ip_feature_cols)
    df_train_features.head()

    # Process data with pipeline
    pipeline = construct_pipeline(numeric_cols, categorical_cols)
    X_train = pipeline.fit_transform(df_train_features)
    y_train = df_train["attack"]
    X_test = pipeline.transform(df_test_features)
    y_test = df_test["attack"]
    print("--- Training Data ---")
    print("Feature matrix shape:", X_train.shape)
    print("Labels shape:", y_train.shape)
    print("--- Test Data ---")
    print("Feature matrix shape:", X_test.shape)
    print("Labels shape:", y_test.shape)

    # Potentially save processed data to disk for later use