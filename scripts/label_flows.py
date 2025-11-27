import sys
import pytz
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta


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

        # ---- Timestamp parsing ----
        date_str = alert.findtext(".//Time/date")
        time_str = alert.findtext(".//Time/time")
        duration_str = alert.findtext(".//Time/sessionduration")

        # Convert date+time to datetime
        start_dt = datetime.strptime(
            f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S"
        )

        # Parse HH:MM:SS duration
        h, m, s = map(int, duration_str.split(":"))
        duration = timedelta(hours=h, minutes=m, seconds=s)
        end_dt = start_dt + duration

        # ---- IPs ----
        src_ip = alert.findtext(".//Source//address")
        dst_ip = alert.findtext(".//Target//address")

        # ---- Ports ----
        sport = alert.findtext(".//Target/Service/sport")
        dport = alert.findtext(".//Target/Service/dport")

        alerts.append({
            "phase": phase,
            "start_time": start_dt,
            "end_time": end_dt,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "sport": int(sport) if sport else None,
            "dport": int(dport) if dport else None,
            "attack": 1
        })

    return pd.DataFrame(alerts)


def merge_flows_labels(flows_df, labels_df):
    '''
    Merge Zeek flows with IDMEF labels based on 5-tuple and time overlap.
    Args:
        flows_df (pd.DataFrame): DataFrame containing Zeek flow data.
        labels_df (pd.DataFrame): DataFrame containing IDMEF alert data.
    Returns:
        pd.DataFrame: Merged DataFrame with attack labels.
    '''
    flows_df = flows_df.copy()

    # Treat flows as UTC
    flows_df["start_time"] = pd.to_datetime(flows_df["start_time"], unit="s", utc=True)
    flows_df["end_time"] = pd.to_datetime(flows_df["end_time"], unit="s", utc=True)

    # Labels also in UTC
    labels_df["start_time"] = pd.to_datetime(labels_df["start_time"]).dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    labels_df["end_time"] = pd.to_datetime(labels_df["end_time"]).dt.tz_localize("US/Eastern").dt.tz_convert("UTC")

    print(flows_df.head())
    print(labels_df.head())

    # Ensure ports are numeric for matching
    flows_df["sport"] = pd.to_numeric(flows_df["sport"], errors="coerce")
    flows_df["dport"] = pd.to_numeric(flows_df["dport"], errors="coerce")
    labels_df["sport"] = pd.to_numeric(labels_df["sport"], errors="coerce")
    labels_df["dport"] = pd.to_numeric(labels_df["dport"], errors="coerce")

    # Initialize label columns
    flows_df["attack"] = 0
    flows_df["phase"] = 0 # default phase 0 (no attack)

    # Merge on 2-tuple (src_ip, dst_ip) first to reduce comparisons
    merged = flows_df.merge(
        labels_df,
        left_on=["src_ip", "dst_ip"],
        right_on=["src_ip", "dst_ip"],
        how="left",
        suffixes=("", "_label")
    )

    # Assign attack only if either:
    # 1) Ports match, or 2) label has no ports
    mask = (
        ( (merged["sport"] == merged["sport_label"]) | merged["sport_label"].isna() ) &
        ( (merged["dport"] == merged["dport_label"]) | merged["dport_label"].isna() ) &
        (merged["start_time"] >= merged["start_time_label"]) &
        (merged["start_time"] <= merged["end_time_label"])
    )

    merged.loc[mask, "attack"] = 1
    merged.loc[mask, "phase"] = merged.loc[mask, "phase_label"]

    # Drop extra label columns
    merged = merged[flows_df.columns.tolist()]

    # Sort by start_time
    merged = merged.sort_values("start_time").reset_index(drop=True)

    return merged


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

    # for phase in range(3,4):
    for phase in range(1, 6):
        print(f"Processing Phase {phase} ...")

        # Load flows
        flows_file = flows_dir / f"phase{phase}_flows.csv"
        flows_df = pd.read_csv(flows_file)

        # Load XML labels
        xml_file = labels_dir / f"mid-level-phase-{phase}.xml"
        labels_df = parse_idmef_xml(xml_file, phase=phase)

        # Merge
        labeled = merge_flows_labels(flows_df, labels_df)
        all_labeled.append(labeled)

    final_df = pd.concat(all_labeled, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved dataset to: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python label_flows.py <dataset_directory>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    dataset_type = dataset_dir.name.lower()  # should be 'inside' or 'dmz'

    # Automatically set the flows directory
    flows_directory = dataset_dir / "flows"

    # Automatically set the labels directory
    labels_directory = dataset_dir / "labels"

    # Automatically set the output CSV
    output_csv_path = dataset_dir / f"{dataset_type}_labeled_flows.csv"

    # Call your main function
    build_dataset(flows_directory, labels_directory, output_csv_path)