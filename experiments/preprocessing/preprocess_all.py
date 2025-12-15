import argparse
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from helper_func.data_split_func import stratified_split, host_temporal_split
from helper_func.preprocess_func import preprocess_data
from helper_func.sampling_func import sample_classes_random, sample_per_phase


def load_config(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)
    

def process_per_phase_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir):
    for phase in range(1,6):
        print(f"\nProcessing phase {phase} data...")
        df_train_per_phase = df_train.copy()
        df_test_per_phase = df_test.copy()

        # Add phase label column
        label_name = f"is_phase_{phase}"
        df_train_per_phase[label_name] = (df_train_per_phase['phase'] == phase).astype(int)
        df_test_per_phase[label_name] = (df_test_per_phase['phase'] == phase).astype(int)
        
        if sampling:
            desired_target = int(sampling.get("per_phase").get("desired_target"))
            df_train_per_phase = sample_per_phase(
                df_train_per_phase, label_name, phase, desired_target
            )

        preprocess_data(
            df_train_per_phase, df_test_per_phase, feature_list, ip_encoding,
            output_dir = out_dir / f"phase_{phase}",
            label_name = label_name
        )


def process_all_phases_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir):
    print("\nProcessing all-phases data...")
    
    # Apply sampling if specified
    if sampling:
        # Extract sampling parameters
        for strategy in sampling.get("all_phases"):
            mode = strategy.get("mode")  # 'upsample' or 'downsample'
            desired_target = int(strategy.get("desired_target"))

            print(f"Applying sampling: mode={mode}, desired_target={desired_target}")
            df_train_sampled, _ = sample_classes_random(mode, df_train, df_train["phase"], desired_target)
        
            # Preprocess and save
            preprocess_data(
                df_train_sampled, df_test, feature_list, ip_encoding,
                output_dir = out_dir / f"{mode}"
            )
    else:
        # Preprocess and save
        preprocess_data(
            df_train, df_test, feature_list, ip_encoding,
            output_dir = out_dir / "all_phases"
        )
    

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

    # Extract job parameters
    feature_list = job.get("feature_list")
    ip_encoding = job.get("ip_encoding")
    split = job.get("split")
    per_phase = job.get("per_phase")
    sampling = job.get("sampling")
    out_dir = base_experiments / job.get("output_dir")

    if dry_run:
        print(f"Dry run: would preprocess with features={feature_list} and ip_encoding={ip_encoding}")
        print(f"Dry run: would perform split: {split}")
        print(f"Dry run: per_phase={per_phase}")
        print(f"Dry run: would apply sampling: {sampling}")
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

    # Process data, both all-phases and per-phase
    process_all_phases_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir)
    process_per_phase_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir)

    
def main():
    # Command: uv run python experiments/preprocessing/preprocess_all.py --config setting.json [--list] [--job job_name] [--dry-run]

    ap = argparse.ArgumentParser(description="Run preprocessing jobs defined in a JSON config")
    ap.add_argument("--config", default="setting.json", help="Path to JSON config (relative to experiments/preprocessing)")
    ap.add_argument("--list", action="store_true", help="Print available jobs in the config and exit")
    ap.add_argument("--job", help="Run a specific job by name (multiple allowed, comma-separated)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
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
