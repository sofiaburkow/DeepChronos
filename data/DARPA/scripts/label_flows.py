import hashlib
import pandas as pd
import argparse
from pathlib import Path


def create_hash(df):
    """Create a hash for each flow based on its features (excluding flow_id)."""
    return (
        df.drop(columns=["flow_id"])
        .astype(str)
        .agg("|".join, axis=1)
        .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    )


def label_flows(flows_dir, attack_flows_dir, output_dir):
    """Label flows in the full dataset based on the attack flow datasets for each phase."""

    # Load full dataset
    all_flows_file = Path(flows_dir) / "flows.csv"
    df_all = pd.read_csv(all_flows_file)
    df_all["flow_hash"] = create_hash(df_all)

    attack_dfs = []

    # Load and prepare all phase datasets
    for phase in range(1, 6):
        attack_flows_file = Path(attack_flows_dir) / f"phase{phase}_flows.csv"
        df_attack = pd.read_csv(attack_flows_file)

        df_attack["flow_hash"] = create_hash(df_attack)
        df_attack["phase"] = phase

        attack_dfs.append(df_attack[["flow_hash", "phase"]])

    # Concatenate all attack phases
    df_attack_all = pd.concat(attack_dfs, ignore_index=True)

    # Merge once
    df_all = df_all.merge(
        df_attack_all,
        on="flow_hash",
        how="left"
    )

    # Fill benign flows
    df_all["phase"] = df_all["phase"].fillna(0).astype(int)

    # Remove hash column
    df_all.drop(columns=["flow_hash"], inplace=True)

    # Save result
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(output_dir / "all_flows_labeled.csv", index=False)

    print("Labeling complete.")
    print(df_all["phase"].value_counts())


if __name__ == "__main__":
    # Command: uv run python data/DARPA/scripts/label_flows.py --scenario one --network inside

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="one", choices=["one", "two"], help="Which DARPA scenario to process")
    ap.add_argument("--network", default="inside", choices=["inside", "dmz"], help="Which network segment to process")
    args = ap.parse_args()

    flows_dir = f"data/DARPA/scenario_{args.scenario}/{args.network}/all_flows"
    attack_flows_dir = f"data/DARPA/scenario_{args.scenario}/{args.network}/per_phase_flows"
    output_dir = f"data/DARPA/scenario_{args.scenario}/{args.network}/labeled_flows"

    label_flows(
        flows_dir=flows_dir,
        attack_flows_dir=attack_flows_dir,
        output_dir=output_dir
    )