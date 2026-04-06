import hashlib
import pandas as pd
import argparse
from pathlib import Path
import re


def clean_tstat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert:
        '#15#c_ip:1' -> 'c_ip'
        'first:29'   -> 'first'
    """

    new_cols = {}

    for col in df.columns:
        # remove "#number#" at beginning
        col_clean = re.sub(r"^#\d+#", "", col)

        # remove ":number" at end
        col_clean = re.sub(r":\d+$", "", col_clean)

        new_cols[col] = col_clean

    df = df.rename(columns=new_cols)

    return df


def rename_tstat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Tstat schema to match Zeek schema for easier merging."""

    df = df.rename(columns={
        "c_ip": "src_ip",
        "s_ip": "dst_ip",
        "c_port": "sport",
        "s_port": "dport",
        "first": "start_time",
        "last": "end_time",
    })

    return df


def compute_hash(df: pd.DataFrame, columns: list) -> pd.Series:
    return (
        df[columns]
        .astype(str)
        .agg("|".join, axis=1)
        .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    )


def label_darpa_flows(
        flows_dir: Path, 
        out_dir: Path, 
        overwrite: bool
        ):

    # Load full flow dataset (which contains all flows, both attack and benign)
    all_flows_file = flows_dir / "all_flows.csv"
    if not all_flows_file.exists():
        raise FileNotFoundError(f"{all_flows_file} not found")

    print("[+] Loading unlabeled flows...")
    df_all = pd.read_csv(all_flows_file)
    match_columns = [col for col in df_all.columns if col != "flow_id"]
    df_all["flow_hash"] = compute_hash(df_all, match_columns)

    # Load per-phase attack flow datasets, compute hashes, and combine into one DataFrame
    print("[+] Loading per-phase attack flows...")
    attack_dfs = []
    for phase in range(1, 6):
        phase_file = flows_dir / f"phase{phase}_flows.csv"
        if not phase_file.exists():
            print(f"[!] Warning: {phase_file.name} not found")
            continue

        df_attack = pd.read_csv(phase_file)
        df_attack["flow_hash"] = compute_hash(df_attack, match_columns)
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

    output_file = out_dir / "all_flows_labeled.csv"
    if output_file.exists() and not overwrite:
        print(f"[!] {output_file.name} already exists. Use --overwrite to replace.")
        return

    df_all.to_csv(output_file, index=False)

    print("[✓] Labeling complete.")
    print(df_all["phase"].value_counts())


def label_ait_flows(
        flows_dir: Path,
        labels_dir: Path,
        out_dir: Path, 
        overwrite: bool
    ):
    
    print("[+] Loading labeled netflows...")
    labels_path = labels_dir / f"all_netflows.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} not found")

    df_attack = pd.read_csv(labels_path)
    match_columns = ["src_ip", "dst_ip", "sport", "dport", "start_time_match", "end_time_match"]
    df_attack["flow_hash"] = compute_hash(
        df_attack,
        match_columns
    )
    
    sensor_hosts = ["cloud_share", "inet_firewall", "internal_share", "intranet_server", "mail", "vpn", "webserver"]
    dfs = []
    for sensor_host in sensor_hosts:
        print(f"\n=== Processing sensor host: {sensor_host} ===")

        # Load unlabeled flows for this sensor host
        flows_unlabeled_file = flows_dir / f"{sensor_host}_flows.csv"
        if not flows_unlabeled_file.exists():
            raise FileNotFoundError(f"{flows_unlabeled_file} not found")
        

        print(f"[+] Loading unlabeled Zeek flows...")
        df_unlabeled = pd.read_csv(flows_unlabeled_file)

        df_unlabeled["start_time_match"] = df_unlabeled["start_time"].round(1)
        df_unlabeled["end_time_match"] = df_unlabeled["end_time"].round(1)

        df_unlabeled["flow_hash"] = compute_hash(
            df_unlabeled,
            match_columns
        )

        print("[+] Labeling Zeek flows...")
        df_labeled = df_unlabeled.merge(
            df_attack[["flow_hash", "label"]],
            on="flow_hash",
            how="left"
        )

        df_labeled["label"] = df_labeled["label"].fillna("benign")
        df_labeled.drop(
            # columns=["flow_hash", "start_time_match", "end_time_match"], 
            columns=["start_time_match", "end_time_match"], 
            inplace=True
        )

        output_file = out_dir / f"{sensor_host}_labeled.csv"
        if output_file.exists() and not overwrite:
            print(f"[!] {output_file.name} already exists. Use --overwrite to replace.")
            return
        
        df_labeled.to_csv(output_file, index=False)

        print("[✓] Labeling complete.")
        print(df_labeled["label"].value_counts())

        df_labeled["sensor_host"] = sensor_host
        dfs.append(df_labeled)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("start_time").reset_index(drop=True)
    output_file = out_dir / f"all_labeled.csv"
    df_all.to_csv(output_file, index=False)


def main(dataset: str, scenario: str, overwrite: bool):
    
    base_dir = Path("data/interim") / dataset / scenario
    flows_dir = base_dir / "flows_unlabeled"
    flows_labeled_dir = base_dir / "flows_labeled"
    flows_labeled_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "darpa2000":
        label_darpa_flows(
            flows_dir=flows_dir, 
            out_dir=flows_labeled_dir, 
            overwrite=overwrite
            )
    elif dataset == "aitv2":
        labels_dir = Path(f"data/interim/{dataset}/{scenario}/labels")
        label_ait_flows(
            flows_dir=flows_dir, 
            labels_dir=labels_dir, 
            out_dir=flows_labeled_dir, 
            overwrite=overwrite
            )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    # uv run python -m src.flow_processing.label_flows --dataset aitv2 --scenario fox --overwrite

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Example: darpa2000")
    parser.add_argument("--scenario", required=True, help="Example: s1_inside")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labeled files")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        scenario=args.scenario,
        overwrite=args.overwrite
    )