from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict, deque


WINDOW_SECONDS = 60


def behavioral_features(df, window_seconds=WINDOW_SECONDS):

    df = df.sort_values("start_time").reset_index(drop=True)

    src_flow_queue = defaultdict(deque)
    dst_flow_queue = defaultdict(deque)

    connections_per_src_list = []
    unique_targets_list = []
    unique_dports_list = []
    syn_failure_ratio_list = []
    reject_ratio_list = []
    reset_ratio_list = []

    connections_per_dst_list = []
    unique_sources_list = []

    for _, row in df.iterrows():

        t = float(row["start_time"])
        src = row["src_ip"]
        dst = row["dst_ip"]
        dport = row["dport"]
        conn_state = str(row.get("conn_state", ""))

        # === Source-centric features ===
        src_flow_q = src_flow_queue[src]
        while src_flow_q and t - src_flow_q[0][0] > window_seconds:
            src_flow_q.popleft()
        # Add current flow
        src_flow_q.append((t, dst, dport, conn_state))

        # === Destination-centric features ===
        dst_flow_q = dst_flow_queue[dst]
        while dst_flow_q and t - dst_flow_q[0][0] > window_seconds:
            dst_flow_q.popleft()
        dst_flow_q.append((t, src))

        # -------- Source features --------
        connections_per_src = len(src_flow_q)
        unique_targets = len({d for _, d, _, _ in src_flow_q})
        unique_dports = len({dp for _, _, dp, _ in src_flow_q})

        syn_failure_ratio = (
            sum(1 for _, _, _, s in src_flow_q if s == "S0") / connections_per_src
            if connections_per_src > 0
            else 0.0
        )
        reject_ratio = (
            sum(1 for _, _, _, s in src_flow_q if s == "REJ") / connections_per_src
            if connections_per_src > 0
            else 0.0
        )
        reset_ratio = (
            sum(1 for _, _, _, s in src_flow_q if s == "RSTO") / connections_per_src
            if connections_per_src > 0
            else 0.0
        )
        
        # -------- Destination features --------
        connections_per_dst = len(dst_flow_q)
        unique_sources = len({s for _, s in dst_flow_q})

        # Append features to lists
        connections_per_src_list.append(connections_per_src)
        unique_targets_list.append(unique_targets)
        unique_dports_list.append(unique_dports)
        syn_failure_ratio_list.append(syn_failure_ratio)
        reject_ratio_list.append(reject_ratio)
        reset_ratio_list.append(reset_ratio)
        connections_per_dst_list.append(connections_per_dst)
        unique_sources_list.append(unique_sources)

    df["connections_per_src_60s"] = connections_per_src_list
    df["unique_targets_60s"] = unique_targets_list
    df["unique_dports_60s"] = unique_dports_list
    df["syn_failure_ratio_60s"] = syn_failure_ratio_list
    df["reject_ratio_60s"] = reject_ratio_list
    df["reset_ratio_60s"] = reset_ratio_list
    df["connections_per_dst_60s"] = connections_per_dst_list
    df["unique_sources_per_dst_60s"] = unique_sources_list

    return df


def main(input_path, output_path):
    print(f"[+] Loading {input_path}")
    df = pd.read_csv(input_path)
    df_behav = behavioral_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_behav.to_csv(output_path, index=False)
    print(f"[✓] Saved behavioral dataset to {output_path}")

# uv run python -m src.feature_engineering.gen_behavioral_features --dataset darpa2000 --scenario s1_dmz
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    args = parser.parse_args()

    input_path = Path(
        f"data/interim/{args.dataset}/{args.scenario}/flows_labeled/all_flows_labeled.csv"
    )
    output_path = Path(
        f"data/interim/{args.dataset}/{args.scenario}/flows_labeled/all_flows_behavioral.csv"
    )

    main(input_path, output_path)