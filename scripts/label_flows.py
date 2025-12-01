import sys
import pytz
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

def parse_idmef_xml(xml_path, phase):
    '''
    Parse a DARPA 2000 IDMEF XML file.
    Args:
        xml_path (str): Path to the IDMEF XML file.
        phase (int): Phase number of the attack.
    Returns:
        pd.DataFrame: DataFrame containing parsed alerts.
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    alerts = []

    for alert in root.findall(".//Alert"):

        alert_id = alert.get("alertid")

        # ---- Timestamp parsing ----
        date_str = alert.findtext(".//Time/date")
        time_str = alert.findtext(".//Time/time")
        duration_str = alert.findtext(".//Time/sessionduration")

        # From US/Eastern local time -> UTC
        eastern = pytz.timezone("US/Eastern")
        start_dt = eastern.localize(
            datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")
        ).astimezone(pytz.UTC)

        start_ts = start_dt.timestamp()

        h, m, s = map(int, duration_str.split(":"))
        end_ts = start_ts + h*3600 + m*60 + s

        # ---- IPs ----
        src_ip = alert.findtext(".//Source//address")
        dst_ip = alert.findtext(".//Target//address")

        # ---- Ports ----
        sport = alert.findtext(".//Target/Service/sport")
        dport = alert.findtext(".//Target/Service/dport")

        alerts.append({
            "alert_id": int(alert_id),
            "alert": 1,
            "phase": phase,
            "start_time": start_ts,
            "duration": end_ts - start_ts,
            "end_time": end_ts,
            "src_ip": src_ip,
            "sport": int(sport) if sport else 0,
            "dst_ip": dst_ip,
            "dport": int(dport) if dport else 0,
        })

    return pd.DataFrame(alerts)


def merge_flows_labels(flows_df, labels_df):
    '''
    Merge Zeek flows with IDMEF labels based on 4-tuple (src_ip, dst_ip, src_port, dst_port) and time overlap.
    Args:
        flows_df (pd.DataFrame): DataFrame containing Zeek flow data.
        labels_df (pd.DataFrame): DataFrame containing IDMEF alert data.
    Returns:
        pd.DataFrame: Merged DataFrame with attack labels.
    '''

    flows = flows_df.copy()
    labels = labels_df.copy()
    
    # Add explicit flow index so we can reduce back to one row per original flow
    flows_with_idx = flows.reset_index().rename(columns={"index": "_flow_idx"})
    
    # Cross-join flows and labels, then compute matching mask on the merged table
    merged = flows_with_idx.merge(labels, how="cross", suffixes=("", "_label"))

    # ==== Matching criteria ====
    
    # IP matching: same or reverse
    same_direction = (
        (merged["src_ip"] == merged["src_ip_label"]) &
        (merged["dst_ip"] == merged["dst_ip_label"]) 
    )
    reverse_direction = (
        (merged["src_ip"] == merged["dst_ip_label"]) &
        (merged["dst_ip"] == merged["src_ip_label"]) 
    )
    
    # Port matching
    icmp_match = (
        (merged["proto"] == "icmp") &
        (merged["proto_label"] == "icmp")
    )
    port_same_dir = ((merged["sport"] == merged["sport_label"]) & (merged["dport"] == merged["dport_label"])) | icmp_match
    port_reverse_dir = ((merged["dport"] == merged["sport_label"]) & (merged["sport"] == merged["dport_label"])) | icmp_match
    
    # Time overlap
    time_match = (
        (np.floor(merged["start_time"]) == np.floor(merged["start_time_label"]))
    )

    # Final matching
    mask = ((same_direction & port_same_dir) | (reverse_direction & port_reverse_dir)) & time_match

    # Initialize label columns in merged
    merged["attack_id"] = 0 # default no alert
    merged["attack"] = 0   # default no attack  
    merged["attack_phase"] = 0    # default phase 0 (no attack)

    # Apply matches
    merged.loc[mask, "attack_id"] = merged.loc[mask, "alert_id"]
    merged.loc[mask, "attack"] = merged.loc[mask, "alert"]
    merged.loc[mask, "attack_phase"] = merged.loc[mask, "phase"]

    # For each original flow, keep a single row. Prefer rows where attack==1.
    merged = merged.sort_values(["_flow_idx", "attack"], ascending=[True, False])
    dedup = merged.drop_duplicates(subset=["_flow_idx"], keep="first")

    # Restore original flow columns order
    flow_columns = flows_with_idx.columns.tolist()
    labeled_flows = dedup[flow_columns + ["attack_id", "attack", "attack_phase"]].copy()

    # Drop the helper _flow_idx column and reset index
    labeled_flows = labeled_flows.drop(columns=["_flow_idx"]).reset_index(drop=True)

    return labeled_flows


def build_dataset(flows_dir, labels_dir, out_csv):
    '''
    Build the labeled dataset by merging Zeek flows with IDMEF labels.
    Args:
        flows_dir (Path): Directory containing Zeek flow CSVs.
        labels_dir (Path): Directory containing IDMEF XML labels.
        out_csv (Path): Path to save the final labeled dataset CSV.
    Returns:
        None
    '''
    all_labeled = []
    for phase in range(1,5): # Not possible to process phase 5 in notebook  
        print(f"Processing Phase {phase} ...")

        # Load flows
        flows_file = f"{flows_dir}/phase{phase}_flows.csv"
        flows_df = pd.read_csv(flows_file)

        # Load XML labels
        xml_file = f"{labels_dir}/mid-level-phase-{phase}.xml"
        labels_df = parse_idmef_xml(xml_file, phase)

        # Merge
        labeled = merge_flows_labels(flows_df, labels_df)
        all_labeled.append(labeled)

    print("Finished processing all phases. Combining and saving dataset...")
    final_df = pd.concat(all_labeled, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved dataset to: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python label_flows.py <dataset_directory>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    dataset_type = dataset_dir.name.lower()  # should be 'inside' or 'dmz'

    # Define paths
    flows_directory = dataset_dir / "flows"
    labels_directory = dataset_dir / "labels"

    # Automatically set the output CSV
    output_csv_path = dataset_dir / f"{dataset_type}_labeled_flows.csv"

    # Build the dataset
    build_dataset(flows_directory, labels_directory, output_csv_path)