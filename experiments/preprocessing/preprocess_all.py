import argparse
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from helper_func.data_split_func import stratified_split, host_temporal_split
from helper_func.preprocess_func import preprocess_data
from helper_func.sampling_func import sample_classes_random


def load_config(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def ensure_out_dir(base_experiments: Path, out_dir: str) -> str:
    # normalize relative paths: if out_dir is relative, interpret relative to experiments folder
    out_path = base_experiments / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    return str(out_path)


def run_job(job: Dict, base_experiments: Path, seed: int = 123, dry_run: bool = False):
    name = job.get("name")
    print(f"\n=== Running job: {name} ===")

    input_csv = job.get("input_csv")
    if not input_csv:
        raise ValueError("job missing input_csv")

    # Resolve csv path relative to this file
    csv_path = Path(__file__).parent.joinpath(input_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    feature_set = job.get("feature_set")
    ip_encoding = job.get("ip_encoding")
    split = job.get("split")
    label_name = job.get("label_name")
    out_dir = job.get("output_dir")

    full_out_dir = ensure_out_dir(base_experiments, out_dir)

    if dry_run:
        print(f"Dry run: would preprocess to {full_out_dir} with features={feature_set} ip_encoding={ip_encoding}")
        print(f"Dry run: would perform split: {split}")
        print(f"Dry run: would apply sampling: {job.get('sampling')}")
        return

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Perform split
    split_type = split.get("type")
    if split_type == "stratified":
        train_size = float(split.get("train_size"))
        stratify_col = split.get("by")
        df_train, df_test = stratified_split(df, train_size, stratify_col)
    elif split_type == "host_temporal":
        train_ratio = float(split.get("train_size"))
        df_train, df_test = host_temporal_split(df, train_ratio)
    else:
        raise ValueError(f"Unknown split type: {split}")
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Apply sampling if specified
    sampling = job.get("sampling")
    if sampling:
        mode = sampling.get("mode")  # 'upsample' or 'downsample'
        desired_target = int(sampling.get("desired_target"))

        print(f"Applying sampling: mode={mode}, desired_target={desired_target}")
        df_train_sampled, _ = sample_classes_random(mode, df_train, df_train["phase"], desired_target)
        df_train = df_train_sampled.copy()

    # Preprocess and save
    preprocess_data(df_train, df_test, feature_set, ip_encoding, full_out_dir, label_name)


def main():
    # Command: uv run python experiments/preprocessing/preprocess_all.py --config setting.json [--list] [--job job_name] [--dry-run]

    ap = argparse.ArgumentParser(description="Run preprocessing jobs defined in a JSON config")
    ap.add_argument("--config", default="setting.json", help="Path to JSON config (relative to experiments/preprocessing)")
    ap.add_argument("--list", action="store_true", help="Print available jobs in the config and exit")
    ap.add_argument("--job", help="Run a specific job by name (multiple allowed, comma-separated)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without running preprocessing")
    args = ap.parse_args()

    base = Path(__file__).parent
    config_path = base / args.config

    # Load config file with jobs to run 
    cfg = load_config(str(config_path))
    jobs = cfg.get("jobs", [])
    if args.list:
        print("Jobs:")
        for j in jobs:
            print(" -", j.get("name"))
        return

    # Option to only run specific jobs
    selected = None
    if args.job:
        wanted = [s.strip() for s in args.job.split(",")]
        selected = [j for j in jobs if j.get("name") in wanted]
        if not selected:
            print("No matching jobs found for:", wanted)
            return
    else:
        selected = jobs

    # Run selected jobs
    for job in selected:
        run_job(job, base_experiments=base, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
