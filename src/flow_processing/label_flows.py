import hashlib
import pandas as pd
import argparse
from pathlib import Path
import re


def clean_tstat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert:
        '#15#c_ip    -> c_ip'
        '#c_ip       -> c_ip'
        'first:29'   -> 'first'
    """

    new_cols = {}

    for col in df.columns:
        # remove "#number#" at beginning
        col_clean = re.sub(r"^(?:#\d+#|#)?|:\d+$", "", col)

        # remove ":number" at end
        col_clean = re.sub(r":\d+$", "", col_clean)

        new_cols[col] = col_clean

    df = df.rename(columns=new_cols)

    # print(f"[+] Cleaned columns: {list(df.columns)}")

    return df


def rename_tstat_columns(df):

    df_renamed = df.rename(columns={
        "c_ip": "src_ip",
        "s_ip": "dst_ip",
        "c_port": "sport",
        "s_port": "dport",
    })

    return df_renamed


def normalize_tstat_times(df):
    """
    Create unified start_time and end_time columns (seconds).
    Works for both TCP and UDP exports.
    """

    # ---------- TCP ----------
    if "first" in df.columns:
        df["start_time"] = pd.to_numeric(df["first"], errors="coerce") / 1000
        df["end_time"]   = pd.to_numeric(df["last"], errors="coerce") / 1000
        return df

    # ---------- UDP ----------
    if "c_first_abs" in df.columns:
        c_start = pd.to_numeric(df["c_first_abs"], errors="coerce")
        s_start = pd.to_numeric(df.get("s_first_abs"), errors="coerce")

        # safest start estimate
        df["start_time"] = pd.concat([c_start, s_start], axis=1).min(axis=1) / 1000

        c_dur = pd.to_numeric(df.get("c_durat"), errors="coerce").fillna(0)
        s_dur = pd.to_numeric(df.get("s_durat"), errors="coerce").fillna(0)

        duration = pd.concat([c_dur, s_dur], axis=1).max(axis=1)

        df["end_time"] = df["start_time"] + duration

        return df

    raise ValueError("Unknown Tstat schema — no timestamp columns found")


def compute_hash(df: pd.DataFrame, columns: list) -> pd.Series:
    return (
        df[columns]
        .astype(str)
        .agg("|".join, axis=1)
        .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    )


def build_labels(labels_dir: Path, base_cols: list, output_dir: Path):

    label_files = [
        "tcp_complete.csv",
        "tcp_nocomplete.csv",
        "udp_complete.csv",
    ]

    dfs = []

    print("[+] Loading label files...")
    for f in label_files:
        path = labels_dir / f
        print("   ", path.name)

        df = pd.read_csv(path)
        df = clean_tstat_columns(df)
        df = rename_tstat_columns(df)
        df = normalize_tstat_times(df)
         
        df["start_time_match"] = df["start_time"].round(0)
        df["end_time_match"] = df["end_time"].round(0)

        dfs.append(df[[
            "src_ip", 
            "dst_ip", 
            "sport", 
            "dport", 
            "start_time_match", 
            "end_time_match",
            "label"
        ]])

    df_all = pd.concat(dfs, ignore_index=True)

    df_all["flow_hash"] = compute_hash(
        df_all,
        base_cols + ["start_time_match", "end_time_match"]
    )

    df_all["start_hash"] = compute_hash(
        df_all,
        base_cols + ["start_time_match"]
    )

    df_all["end_hash"] = compute_hash(
        df_all,
        base_cols + ["end_time_match"]
    )

    df_all = df_all[
        base_cols +
        ["start_time_match", "end_time_match", "flow_hash", "start_hash", "end_hash", "label"]
    ]  

    output_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_dir / "all_netflows.csv", index=False) 

    return df_all


def label_ait_flows(
        flows_dir: Path,
        labels_dir: Path,
        out_dir: Path, 
        overwrite: bool
    ):

    base_cols = ["src_ip", "dst_ip", "sport", "dport"]

    print("[+] Processing labeled netflows...")
    df_labels = build_labels(
        labels_dir=labels_dir, 
        base_cols=base_cols,
        output_dir=labels_dir
    )
    
    start_label_map = (
        df_labels
        .drop_duplicates("start_hash")
        .set_index("start_hash")["label"]
    )

    end_label_map = (
        df_labels
        .drop_duplicates("end_hash")
        .set_index("end_hash")["label"]
    )
    
    sensor_hosts = ["cloud_share", "inet_firewall", "internal_share", "intranet_server", "mail", "vpn", "webserver", "attacker_0"]
    dfs = []
    for sensor_host in sensor_hosts:
        print(f"\n=== Processing sensor host: {sensor_host} ===")

        # Load unlabeled flows for this sensor host
        flows_unlabeled_file = flows_dir / f"{sensor_host}_flows.csv"
        if not flows_unlabeled_file.exists():
            raise FileNotFoundError(f"{flows_unlabeled_file} not found")

        print(f"[+] Loading unlabeled Zeek flows...")
        df_unlabeled = pd.read_csv(flows_unlabeled_file)

        df_unlabeled["start_time_match"] = df_unlabeled["start_time"].round(0)
        df_unlabeled["end_time_match"] = df_unlabeled["end_time"].round(0)

        df_unlabeled["flow_hash"] = compute_hash(
            df_unlabeled,
            base_cols + ["start_time_match", "end_time_match"]
        )

        df_unlabeled["start_hash"] = compute_hash(
            df_unlabeled,
            base_cols + ["start_time_match"]
        )

        df_unlabeled["end_hash"] = compute_hash(
            df_unlabeled,
            base_cols + ["end_time_match"]
        )

        print("[+] Labeling Zeek flows...")

        # Strict matching
        df_labeled = df_unlabeled.merge(
            df_labels[["flow_hash", "label"]],
            on="flow_hash",
            how="left"
        )

        # Fallback logic
        missing_mask = df_labeled["label"].isna()

        # start-time fallback
        df_labeled.loc[missing_mask, "label_start"] = (
            df_labeled.loc[missing_mask, "start_hash"]
            .map(start_label_map)
        )

        # recompute mask after start fallback
        missing_mask = df_labeled["label"].isna()

        # end-time fallback
        df_labeled.loc[missing_mask, "label_end"] = (
            df_labeled.loc[missing_mask, "end_hash"]
            .map(end_label_map)
        )

        # Priority resolution: strict > start fallback > end fallback > benign
        df_labeled["label"] = (
            df_labeled["label"]
            .combine_first(df_labeled["label_start"])
            .combine_first(df_labeled["label_end"])
            .fillna("benign")
        )

        df_labeled.drop(
            columns=[
                "start_time_match",
                "end_time_match",
                "label_start",
                "label_end",
            ],
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
    output_file = out_dir / f"all_flows_labeled_unprocessed.csv"
    df_all.to_csv(output_file, index=False)


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
        # labels_dir = Path(f"data/interim/{dataset}/{scenario}/labels")
        labels_dir = Path(f"data/raw/{args.dataset}/{args.scenario}_netflows")
        label_ait_flows(
            flows_dir=flows_dir, 
            labels_dir=labels_dir, 
            out_dir=flows_labeled_dir, 
            overwrite=overwrite
            )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    # uv run python -m src.flow_processing.label_flows --dataset aitv2 --scenario santos --overwrite

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