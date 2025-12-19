import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict

from helper_func.data_split_func import (
    stratified_split, host_temporal_split, prepare_phase_dataset, 
    build_sequences, temporal_split
)
from helper_func.preprocess_func import (
    prepare_features, construct_pipeline, preprocess_data, save_data
)
from helper_func.sampling_func import (
    sample_classes_random, sample_per_phase
)


def load_config(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def process_per_phase_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir, random_state):
    for phase in range(1, 6):
        print(f"\nProcessing phase {phase} data...")

        label_name = "y"
        df_train_per_phase = prepare_phase_dataset(df_train, phase, label_name=label_name)
        df_test_per_phase = prepare_phase_dataset(df_test, phase, label_name=label_name)
        
        if sampling:
            for strategy in sampling:
                mode = strategy.get("mode")  # 'auto'
                desired_target = int(strategy.get("desired_target"))

                print(f"Applying sampling: mode={mode}, desired_target={desired_target}")
                df_train_per_phase_sampled, _ = sample_per_phase(
                    mode=mode, 
                    X_per_phase=df_train_per_phase, 
                    desired_target=desired_target, 
                    phase=phase, 
                    label_name=label_name, 
                    random_state=random_state
                )

                preprocess_data(
                    df_train_per_phase_sampled, df_test_per_phase, feature_list, ip_encoding,
                    output_dir=out_dir/f"{mode}_phase_{phase}",
                    label_name=label_name
                )
        else:
            preprocess_data(
                df_train_per_phase, df_test_per_phase, feature_list, ip_encoding,
                output_dir=out_dir/f"phase_{phase}",
                label_name=label_name
            )
    

def process_all_phases_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir, random_state):
    print("\nProcessing all-phases data...")
    if sampling:
        for strategy in sampling:
            mode = strategy.get("mode")  # 'upsample' or 'downsample'
            desired_target = int(strategy.get("desired_target"))

            print(f"Applying sampling: mode={mode}, desired_target={desired_target}")
            label_name = "phase"
            df_train_sampled, _ = sample_classes_random(
                mode=mode, 
                X=df_train, 
                desired_target=desired_target, 
                label_name=label_name, 
                phases=[1,2,3,4,5],
                random_state=random_state
            )
        
            preprocess_data(
                df_train_sampled, df_test, feature_list, ip_encoding,
                output_dir=out_dir/f"{mode}_all_phases"
            )
    else:
        preprocess_data(
            df_train, df_test, feature_list, ip_encoding,
            output_dir=out_dir/f"all_phases"
        )


def process_temp_data(df, feature_list, ip_encoding, window_size, train_size, out_dir, random_state):
    for phase in range(1,6):
        print(f"\nProcessing phase {phase} temp data...")
        print(f"Window size: {window_size}")
        df_phase = prepare_phase_dataset(df, phase)
    
        df_phase_features, numeric_cols, categorical_cols, _ = prepare_features(df_phase, feature_list, ip_encoding)
        pipeline = construct_pipeline(numeric_cols, categorical_cols)
        X = pipeline.fit_transform(df_phase_features)

        y = df_phase["y"]
        y_phase = df_phase["phase"]

        X_sequences, y_sequences, y_phase_sequences = build_sequences(X, y, y_phase, window_size)
        X_train, X_test, y_train, y_test, y_phase_train, y_phase_test = temporal_split(
            X=X_sequences, 
            y=y_sequences,
            y_phase=y_phase_sequences,
            train_ratio=train_size, 
            window_size=window_size, 
            random_state=random_state
        )

        save_data(
            X_train, y_train, y_phase_train, 
            X_test, y_test, y_phase_test, 
            pipeline, 
            numeric_cols, categorical_cols, ip_encoding, 
            output_dir=out_dir/f"phase_{phase}",
            sparse=False
        )


def run_job(job, input_csv, features_file, base_experiments, dry_run, random_state):
    name = job.get("name")
    print(f"\n=== Running job: {name} ===")

    if not input_csv:
        raise ValueError("job missing input_csv")

    # Resolve csv path relative to this file
    csv_path = Path(__file__).parent.joinpath(input_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    # Load features
    features_path = Path(__file__).parent.joinpath(features_file).resolve()
    with open(features_path) as f:
        feature_list = json.load(f)
    ip_encoding = "none"

    # Extract job parameters
    split = job.get("split")
    mode = job.get("mode") # 'all_phases' or 'per_phase'
    sampling = job.get("sampling")
    out_dir = base_experiments / job.get("output_dir")

    if dry_run:
        print(f"Dry run: would preprocess with features={feature_list} and ip_encoding={ip_encoding}")
        print(f"Dry run: would perform split: {split}")
        print(f"Dry run: mode={mode}")
        print(f"Dry run: would apply sampling: {sampling}")
        print(f"Dry run: would save to: {out_dir}")
        return

    # Load data
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Perform split
    split_type = split.get("type")
    train_size = float(split.get("train_size"))
    if split_type == "stratified":
        df_train, df_test = stratified_split(
            df=df, 
            train_ratio=train_size, 
            stratify_col="phase", 
            random_state=random_state
        )
    elif split_type == "host_temporal":
        df_train, df_test = host_temporal_split(df, train_size)
    elif split_type == "temporal":
        window_size = int(split.get("window_size"))
        process_temp_data(df, feature_list, ip_encoding, window_size, train_size, out_dir, random_state)
        return # temp split processing done 
    
    else:
        raise ValueError(f"Unknown split type: {split}")
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Process data (for strat and host_temp)
    if mode == "all_phases":
        process_all_phases_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir, random_state)
    elif mode == "per_phase":
        process_per_phase_data(df_train, df_test, feature_list, ip_encoding, sampling, out_dir, random_state)
    else:
        raise ValueError(f"Mode must be all_phases or per_phase.")

    
def main():
    # Command: uv run python experiments/preprocessing/preprocess_all.py

    ap = argparse.ArgumentParser(description="Run preprocessing jobs defined in a JSON config")
    ap.add_argument("--config", default="all_jobs.json", help="Path to JSON config (relative to experiments/preprocessing)")
    ap.add_argument("--list", action="store_true", help="Print available jobs in the config and exit")
    ap.add_argument("--job", help="Run a specific job by name (multiple allowed, comma-separated)")
    ap.add_argument("--input_csv", default="../../data/DARPA_2000/Scenario_One/inside/inside_labeled_flows_all.csv", help="Path to JSON config (relative to experiments/preprocessing)")
    ap.add_argument("--features_file", default="features_list.json", help="Path to JSON config (relative to experiments/preprocessing)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without running preprocessing")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    args = ap.parse_args()
            
    base = Path(__file__).parent
    config_path = base / args.config

    # Load config file with jobs to run 
    cfg = load_config(str(config_path))
    jobs = cfg.get("jobs", [])
    if args.list:
        print("Jobs:")
        for job in jobs:
            print(" -", job.get("name"))
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
        run_job(
            job=job, 
            input_csv=args.input_csv, 
            features_file=args.features_file, 
            base_experiments=base, 
            dry_run=args.dry_run,
            random_state=args.seed
        )


if __name__ == "__main__":
    main()
