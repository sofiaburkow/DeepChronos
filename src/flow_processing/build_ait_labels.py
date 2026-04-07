import pandas as pd
from pathlib import Path
import re
import argparse
import hashlib


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


def build_labels(labels_dir: Path, output_dir: Path):

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

        df["start_time_match"] = df["start_time"].round(1)
        df["end_time_match"] = df["end_time"].round(1)

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

    match_columns = ["src_ip", "dst_ip", "sport", "dport", "start_time_match", "end_time_match"]
    df_all["flow_hash"] = compute_hash(
        df_all,
        match_columns
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_dir / "all_netflows.csv", index=False)

    return df_all


if __name__ == "__main__":
    # uv run python -m src.flow_processing.build_ait_labels --dataset aitv2 --scenario fox --overwrite

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, default="aitv2")
    parser.add_argument("--scenario", required=True, default="fox")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels_dir = Path(f"data/raw/{args.dataset}/{args.scenario}/netflows")
    output_dir = Path(f"data/interim/{args.dataset}/{args.scenario}/labels")

    build_labels(labels_dir, output_dir)