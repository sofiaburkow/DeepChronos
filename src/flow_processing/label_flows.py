import hashlib
import pandas as pd
import argparse
from pathlib import Path


def create_hash(df: pd.DataFrame) -> pd.Series:
    """
    Create a hash for each flow based on all features except flow_id.
    Ensures deterministic matching between full and per-phase files.
    """
    return (
        df.drop(columns=["flow_id"])
        .astype(str)
        .agg("|".join, axis=1)
        .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    )


def label_flows(dataset: str, scenario_network: str, overwrite: bool):
    base_dir = Path("data/interim") / dataset / scenario_network

    flows_unlabeled_dir = base_dir / "flows_unlabeled"
    flows_labeled_dir = base_dir / "flows_labeled"

    all_flows_file = flows_unlabeled_dir / "all_flows.csv"

    if not all_flows_file.exists():
        raise FileNotFoundError(f"{all_flows_file} not found")

    print("[+] Loading full flow dataset...")
    df_all = pd.read_csv(all_flows_file)
    df_all["flow_hash"] = create_hash(df_all)

    attack_dfs = []

    print("[+] Loading per-phase attack flows...")
    for phase in range(1, 6):
        phase_file = flows_unlabeled_dir / f"phase{phase}_flows.csv"

        if not phase_file.exists():
            print(f"[!] Warning: {phase_file.name} not found")
            continue

        df_attack = pd.read_csv(phase_file)
        df_attack["flow_hash"] = create_hash(df_attack)
        df_attack["phase"] = phase

        attack_dfs.append(df_attack[["flow_hash", "phase"]])

    if not attack_dfs:
        raise RuntimeError("No phase files found for labeling.")

    df_attack_all = pd.concat(attack_dfs, ignore_index=True)

    print("[+] Merging phase labels into full dataset...")
    df_all = df_all.merge(
        df_attack_all,
        on="flow_hash",
        how="left"
    )

    df_all["phase"] = df_all["phase"].fillna(0).astype(int)

    df_all.drop(columns=["flow_hash"], inplace=True)

    flows_labeled_dir.mkdir(parents=True, exist_ok=True)

    output_file = flows_labeled_dir / "all_flows_labeled.csv"

    if output_file.exists() and not overwrite:
        print(f"[!] {output_file.name} already exists. Use --overwrite to replace.")
        return

    df_all.to_csv(output_file, index=False)

    print("[✓] Labeling complete.")
    print(df_all["phase"].value_counts())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Example: darpa2000")
    parser.add_argument("--scenario_network", required=True, help="Example: s1_inside")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labeled files")
    args = parser.parse_args()

    label_flows(
        dataset=args.dataset,
        scenario_network=args.scenario_network,
        overwrite=args.overwrite
    )