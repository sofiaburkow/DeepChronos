import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. BUILD FAST INDEX FOR ALERTS
# ---------------------------------------------------------------------------

def build_alert_index(labels_df):
    """
    Create a dictionary mapping (src_ip, dst_ip, sport, dport, second) → alert indices.
    Also adds reverse direction.
    """
    index = defaultdict(list)

    for i, row in labels_df.iterrows():
        key = (
            row["src_ip"],
            row["dst_ip"],
            row["sport"],
            row["dport"],
            row["ts_floor"],
        )
        index[key].append(i)

        # reverse direction (TCP response flows swap ports)
        rev_key = (
            row["dst_ip"],
            row["src_ip"],
            row["dport"],
            row["sport"],
            row["ts_floor"],
        )
        index[rev_key].append(i)

        # ICMP matching (no ports)
        if row["proto"] == "icmp":
            icmp_key = (
                row["src_ip"],
                row["dst_ip"],
                "icmp",
                "icmp",
                row["ts_floor"],
            )
            index[icmp_key].append(i)

            icmp_key_rev = (
                row["dst_ip"],
                row["src_ip"],
                "icmp",
                "icmp",
                row["ts_floor"],
            )
            index[icmp_key_rev].append(i)

    return index


# ---------------------------------------------------------------------------
# 2. LABEL FLOWS USING THE INDEX
# ---------------------------------------------------------------------------

def label_flows(flows_df, labels_df, alert_index):
    attack_id = []
    attack = []
    phase = []

    for idx, flow in flows_df.iterrows():

        ts_sec = int(flow["start_time"])

        if flow["proto"] == "icmp":
            key = (
                flow["src_ip"],
                flow["dst_ip"],
                "icmp",
                "icmp",
                ts_sec
            )
        else:
            key = (
                flow["src_ip"],
                flow["dst_ip"],
                int(flow["sport"]),
                int(flow["dport"]),
                ts_sec
            )

        candidates = alert_index.get(key, [])

        if len(candidates) == 0:
            attack_id.append(0)
            attack.append(0)
            phase.append(0)
            continue

        # choose alert closest in time
        best = sorted(
            candidates,
            key=lambda i: abs(flow["start_time"] - labels_df.loc[i, "start_time"])
        )[0]

        attack_id.append(labels_df.loc[best, "alert_id"])
        attack.append(1)
        phase.append(labels_df.loc[best, "phase"])

    flows_df["attack_id"] = attack_id
    flows_df["attack"] = attack
    flows_df["phase"] = phase

    return flows_df


# ---------------------------------------------------------------------------
# 3. MAIN DATASET BUILDER
# ---------------------------------------------------------------------------

def build_dataset(flows_dir, labels_dir, out_csv):
    all_labeled = []

    for phase in range(1, 6):
        print(f"\n=== Processing Phase {phase} ===")

        flows_file = f"{flows_dir}/phase{phase}_flows.csv"
        xml_file   = f"{labels_dir}/phase{phase}_alerts.csv"

        flows_df = pd.read_csv(flows_file)
        labels_df = pd.read_csv(xml_file)

        print(f"Flows: {len(flows_df)}  | Alerts: {len(labels_df)}")

        alert_index = build_alert_index(labels_df)
        labeled = label_flows(flows_df, labels_df, alert_index)

        all_labeled.append(labeled)

    print("\n=== Saving final combined dataset ===")
    final_df = pd.concat(all_labeled, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    # Command: `uv run scripts/label_flows.py data/DARPA_2000/inside`
    
    if len(sys.argv) < 2:
        print("Usage: python label_flows.py <dataset_directory>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    dataset_type = dataset_dir.name.lower()

    flows_directory  = f"{dataset_dir}/flows"
    labels_directory = f"{dataset_dir}/labels"
    output_csv_path  = f"{dataset_dir}/{dataset_type}_labeled_flows.csv"

    build_dataset(flows_directory, labels_directory, output_csv_path)
