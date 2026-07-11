import random
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.feature_engineering.features import BASE_FEATURES
from src.feature_engineering.utils import (
    sort_by_time,
    filter_features,
    process_features,
    check_phase_coverage,
)
import joblib
import scipy.sparse as sp


def process_data(
        data_path, 
        out_dir,
        train_size,
        seed
    ):

    print(f"[+] Loading labeled dataset from {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    df = pd.read_csv(data_path)
    df = sort_by_time(df)
    df["orig_index"] = df.index

    # --- Process Features (select columns and identify numeric/categorical) ---
    features_unprocessed, numerical_cols, categorical_cols = filter_features(
        df,
        BASE_FEATURES,
    )

    # --- Split Data ---
    print("Raw feature shape:", features_unprocessed.shape)
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_size,
        stratify=df["phase"],
        shuffle=True,
        random_state=seed
    )
 
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    # For Kitsune, training is unsupervised on benign traffic only (phase == 0)
    train_benign_mask = train_df["phase"] == 0
    train_df_benign = train_df[train_benign_mask].reset_index(drop=True)

    # Build raw split dataframes to fit/transform pipeline
    X_all = features_unprocessed
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)

    # Fit preprocessing pipeline only on benign training rows
    X_fit = X_train_raw[train_benign_mask]
    if X_fit.shape[0] == 0:
        raise ValueError("No benign samples found in training split to fit preprocessing pipeline.")

    X_train_proc, pipeline = process_features(X_fit, numerical_cols, categorical_cols)
    X_test_proc = pipeline.transform(X_test_raw)

    # Assign final arrays
    X_train = X_train_proc
    X_test = X_test_proc

    # Phases present in the full dataset (useful for checks)
    phases = sorted(set(df["phase"].tolist()))
    print(f"Phases in dataset: {phases}")

    # Validate phase coverage on test set (train set intentionally may only contain benign)
    check_phase_coverage(train_df_benign["phase"].to_numpy(), "Train set (benign)", expected_phases={0})
    check_phase_coverage(test_df["phase"].to_numpy(), "Test set", expected_phases=phases)

    # --- Save Processed Data ---
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save processed feature matrices and metadata for train and test
    # Handle sparse outputs
    if sp.issparse(X_train):
        sp.save_npz(out_dir / "X_train.npz", X_train)
    else:
        np.save(out_dir / "X_train.npy", np.asarray(X_train))

    np.save(out_dir / "y_train.npy", train_df_benign["phase"].to_numpy())

    if sp.issparse(X_test):
        sp.save_npz(out_dir / "X_test.npz", X_test)
    else:
        np.save(out_dir / "X_test.npy", np.asarray(X_test))

    np.save(out_dir / "y_test.npy", test_df["phase"].to_numpy())

    # Persist preprocessing pipeline
    joblib.dump(pipeline, out_dir / "preprocess_pipeline.joblib")

    print(f"[✓] Saved processed data to {out_dir}")


if __name__ == "__main__":
    # Command: uv run python -m Kitsune.prepare_data --dataset darpa2000 --scenario s1_inside

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aitv2")
    parser.add_argument("--scenario", type=str, default="santos")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_path = Path("data/interim") / args.dataset / args.scenario / "flows_labeled" / "all_flows_statistical.csv"
    train_size = 0.7

    out_dir = Path("data/kitsune") / args.dataset / args.scenario
    process_data(
        data_path=data_path,
        out_dir=out_dir,
        train_size=train_size,
        seed=args.seed
    )