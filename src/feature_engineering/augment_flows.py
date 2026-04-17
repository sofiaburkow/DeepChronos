from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, deque, Counter


from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, deque, Counter


def augment_features(df):

    df = df.sort_values("start_time").reset_index(drop=True)

    # --- Stateful structures ---
    time_window = 60 # seconds

    # --- destination-centric ---
    recent_src_per_dst = defaultdict(lambda: deque())
    src_counts_per_dst = defaultdict(Counter)

    # --- source-centric ---
    recent_dst = defaultdict(lambda: deque())
    dst_counts = defaultdict(Counter)

    src_dst_counts = defaultdict(Counter)
    src_total_counts = defaultdict(int)

    scan_history = defaultdict(lambda: {
        "unique_targets": set(),
        "unique_ports": set(),
        "count": 0
    })

    # --- Feature columns ---
    unique_sources_list = []
    fanin_rate_list = []

    unique_targets_list = []
    fanout_rate_list = []
    dst_ratio_list = []
    unique_ports_list = []
    connection_count_list = []

    for _, row in df.iterrows():
        t = float(row["start_time"])
        src = row["src_ip"]
        dst = row["dst_ip"]
        dport = row["dport"]

        # --- destination view (scan detection) ---
        in_queue = recent_src_per_dst[dst]
        in_counts = src_counts_per_dst[dst]

        while in_queue and t - in_queue[0][0] > time_window:
            _, old_src = in_queue.popleft()
            in_counts[old_src] -= 1
            if in_counts[old_src] == 0:
                del in_counts[old_src]

        in_queue.append((t, src))
        in_counts[src] += 1

        unique_sources = len(in_counts)
        fanin_rate = len(in_queue) / time_window

        # --- source view ---
        out_queue = recent_dst[src]
        out_counts = dst_counts[src]

        while out_queue and t - out_queue[0][0] > time_window:
            _, old_dst = out_queue.popleft()
            out_counts[old_dst] -= 1
            if out_counts[old_dst] == 0:
                del out_counts[old_dst]

        out_queue.append((t, dst))
        out_counts[dst] += 1

        unique_targets = len(out_counts)
        fanout_rate = len(out_queue) / time_window

        # --- Pair ratio (optimized) ---
        src_dst_counts[src][dst] += 1
        src_total_counts[src] += 1
        dst_ratio = src_dst_counts[src][dst] / src_total_counts[src]

        # --- Long-term scan features ---
        hist = scan_history[src]
        hist["unique_targets"].add(dst)
        hist["unique_ports"].add(dport)
        hist["count"] += 1

        # --- Store ---
        unique_sources_list.append(unique_sources)
        fanin_rate_list.append(fanin_rate)

        unique_targets_list.append(unique_targets)
        fanout_rate_list.append(fanout_rate)
        dst_ratio_list.append(dst_ratio)
        unique_ports_list.append(len(hist["unique_ports"]))
        connection_count_list.append(hist["count"])

    # --- Add to dataframe ---
    df["unique_sources"] = unique_sources_list
    df["fanin_rate"] = fanin_rate_list

    df["unique_targets"] = unique_targets_list
    df["fanout_rate"] = fanout_rate_list
    df["dst_ratio"] = dst_ratio_list
    df["unique_ports"] = unique_ports_list
    df["connection_count"] = connection_count_list

    return df


def main(input_path, output_path): 
    print(f"[+] Loading {input_path}") 
    df = pd.read_csv(input_path) 
    df_aug = augment_features(df) 
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    df_aug.to_csv(output_path, index=False) 
    print(f"[✓] Saved augmented dataset to {output_path}")


if __name__ == "__main__":
    # uv run python -m src.feature_engineering.augment_flows --dataset aitv2 --scenario santos

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    parser.add_argument("--file_name", type=str, default="all_flows_labeled.csv")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    input_path = Path(f"data/interim/{args.dataset}/{args.scenario}/flows_labeled/{args.file_name}")
    output_path = Path(f"data/interim/{args.dataset}/{args.scenario}/flows_labeled/flows_augmented.csv")

    main(input_path, output_path)