import sys
import pytz
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
        alert_id = int(alert.get("alertid"))

        # ---- Timestamp ----
        date_str = alert.findtext(".//Time/date")
        time_str = alert.findtext(".//Time/time")
        duration_str = alert.findtext(".//Time/sessionduration")

        # Convert timestamp from US/Eastern → UTC epoch seconds
        eastern = pytz.timezone("US/Eastern")
        dt = eastern.localize(
            datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")
        ).astimezone(pytz.UTC)

        start_time = dt.timestamp()

        h, m, s = map(int, duration_str.split(":"))
        end_time = start_time + h*3600 + m*60 + s
        duration = end_time - start_time

        # ---- IPs ----
        src_ip = alert.findtext(".//Source//address")
        dst_ip = alert.findtext(".//Target//address")

        # ---- Ports ----
        sport = alert.findtext(".//Target/Service/sport")
        dport = alert.findtext(".//Target/Service/dport")

        sport = int(sport) if sport else 0
        dport = int(dport) if dport else 0

        # ---- Protocol ----
        service_name = alert.findtext(".//Target/Service/name") or ""
        service_name = service_name.lower()
        
        if service_name.startswith("icmp"):
            proto = "icmp"
        elif service_name == "tcp":
            proto = "tcp"
        elif service_name == "udp":
            proto = "udp"
        else:
            proto = "other"

        alerts.append({
            "alert_id": alert_id,
            "alert": 1,
            "phase": phase,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "sport": sport,
            "dport": dport,
            "proto": proto,
            "ts_floor": int(start_time)  # integer second
        })

    return pd.DataFrame(alerts)

    
if __name__ == "__main__":
    # Command: `uv run scripts/xml_alerts_to_csv.py data/DARPA_2000/inside/labels`

    if len(sys.argv) < 2:
        print("Usage: python xml_alerts_to_csv.py <path_to_xml_alerts_directory>")
        sys.exit(1)

    xml_dir = Path(sys.argv[1]).resolve()

    all_labeled = []
    # Loop over all XML files
    for phase in range(1, 6):
        print(f"\n=== Processing Phase {phase} ===")

        xml_file = f"{xml_dir}/mid-level-phase-{phase}.xml"
        labels_df = parse_idmef_xml(xml_file, phase)
        all_labeled.append(labels_df)

        print(f"Number of Alerts: {len(labels_df)}")

        # Save per-phase CSV
        print(f"\n=== Saving Phase {phase} Alert ===")
        out_csv = f"{xml_dir}/phase{phase}_alerts.csv"
        labels_df.to_csv(out_csv, index=False)
        print(f"Saved → {out_csv}")

    out_csv = f"{xml_dir}/xml_alerts_combined.csv"
    print("\n=== Saving final combined dataset ===")
    final_df = pd.concat(all_labeled, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
