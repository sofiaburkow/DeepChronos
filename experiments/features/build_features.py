import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump


# ---------------------------------------------------------------------------
# IP ENCODING METHODS
# ---------------------------------------------------------------------------

def encode_ip_none(df):
    """Drop IP address fields completely."""
    return df.drop(columns=["src_ip", "dst_ip"]), None

def encode_ip_integer(df):
    """Convert IPv4 x.x.x.x to an integer: a*256^3 + b*256^2 + c*256 + d."""
    def ip_to_int(ip):
        try:
            a, b, c, d = map(int, ip.split("."))
            return (a << 24) + (b << 16) + (c << 8) + d
        except:
            return 0

    df["src_ip_int"] = df["src_ip"].apply(ip_to_int)
    df["dst_ip_int"] = df["dst_ip"].apply(ip_to_int)

    return df.drop(columns=["src_ip", "dst_ip"]), ["src_ip_int", "dst_ip_int"]


def encode_ip_onehot(df):
    """
    One-hot encoding for IP addresses.
    WARNING: Very large dimensionality. Use only for small subsets.
    """
    df = df.copy()
    return df, ["src_ip", "dst_ip"]


IP_ENCODERS = {
    "none": encode_ip_none,
    "integer": encode_ip_integer,
    "onehot": encode_ip_onehot,
}


# ---------------------------------------------------------------------------
# MAIN FEATURE BUILDER
# ---------------------------------------------------------------------------

def build_feature_matrix(df, ip_encoding="none"):
    """
    Construct ML-ready features from labeled flow data.
    """

    # Label = binary attack classification (1 = attack, 0 = benign)
    y = df["attack"].astype(int)

    # Drop metadata fields
    df = df.drop(columns=["flow_id", "attack_id", "phase", "attack"])

    # ----------------------------------------------------------------------
    # 1) Handle IP address fields
    # ----------------------------------------------------------------------
    if ip_encoding not in IP_ENCODERS:
        raise ValueError(f"Unknown IP encoding: {ip_encoding}")

    df, ip_feature_cols = IP_ENCODERS[ip_encoding](df)

    # ----------------------------------------------------------------------
    # 2) Categorical features
    # ----------------------------------------------------------------------
    categorical_cols = ["proto", "service", "conn_state"]

    # Some columns may not exist (e.g., service = '-' for ICMP)
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # ----------------------------------------------------------------------
    # 3) Numerical features
    # ----------------------------------------------------------------------
    numeric_cols = [
        "duration", "sport", "dport",
        "orig_bytes", "resp_bytes",
        "orig_pkts", "resp_pkts",
    ]

    if ip_feature_cols:
        numeric_cols += ip_feature_cols

    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # ----------------------------------------------------------------------
    # ColumnTransformer: one-hot + normalization
    # ----------------------------------------------------------------------
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    pipeline = Pipeline(steps=[("transform", transformer)])

    X = pipeline.fit_transform(df)

    return X, y, pipeline, numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ML-ready feature matrix.")
    parser.add_argument("--input", required=True, help="Path to labeled CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--ip-encoding", default="none",
                        choices=["none", "integer", "onehot"],
                        help="How to encode IP addresses.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    print("Building features...")
    X, y, pipeline, num_cols, cat_cols = build_feature_matrix(
        df, ip_encoding=args.ip_encoding
    )

    # Save outputs
    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "y.npy"), y)
    dump(pipeline, os.path.join(args.output_dir, "feature_pipeline.joblib"))

    with open(os.path.join(args.output_dir, "feature_info.txt"), "w") as f:
        f.write("Numerical features:\n")
        f.write(str(num_cols) + "\n\n")
        f.write("Categorical features:\n")
        f.write(str(cat_cols) + "\n\n")
        f.write(f"IP encoding: {args.ip_encoding}\n")

    print(f"Saved X, y, and preprocessing pipeline to {args.output_dir}/")


if __name__ == "__main__":
    main()