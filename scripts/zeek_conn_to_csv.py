import sys
from pathlib import Path
import csv

def safe_int(val, default=0):
    '''
    Safely convert a value to int, return default if conversion fails.
    '''
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def safe_float(val, default=0.0):
    '''
    Safely convert a value to float, return default if conversion fails.
    '''
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def parse_conn_log(input_file, output_csv):
    '''
    Parse Zeek conn.log file and convert to CSV format.

    Args:
        input_file (str): Path to the Zeek conn.log file.
        output_csv (str): Path to the output CSV file.
    Returns:
        None
    '''
    with open(input_file) as f, open(output_csv, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow([
            "flow_id",
            "start_time",
            "end_time",
            "duration",
            "src_ip",
            "sport",
            "dst_ip",
            "dport",
            "proto",
            "service",
            "orig_bytes",
            "resp_bytes",
            "conn_state",
            "local_orig",
            "local_resp",
            "missed_bytes",
            "history",
            "orig_pkts",
            "orig_ip_bytes",
            "resp_pkts",
            "resp_ip_bytes",
            "tunnel_parents",
            "ip_proto"  
        ])

        fieldnames = []
        num_header_lines = 0

        for i, line in enumerate(f):
            line = line.strip()

            if not line or line.startswith("#"):
                if line.startswith("#fields"):
                    # extract field names
                    fieldnames = line.split()[1:]
                num_header_lines += 1
                continue

            values = line.split("\t")
            row = dict(zip(fieldnames, values))

            # Extract relevant fields
            flow_id = f"f{i-num_header_lines}"
            ts = safe_float(row.get("ts", 0))
            dur = safe_float(row.get("duration", 0))
            src_ip = row.get("id.orig_h", "") or ""
            dst_ip = row.get("id.resp_h", "") or ""
            src_port = safe_int(row.get("id.orig_p", "0"))
            dst_port = safe_int(row.get("id.resp_p", "0"))

            writer.writerow([
                flow_id,
                ts,
                ts + dur,
                dur,
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                row.get("proto", ""),   
                row.get("service", ""),
                safe_int(row.get("orig_bytes", "0")),
                safe_int(row.get("resp_bytes", "0")),
                row.get("conn_state", ""),
                row.get("local_orig", ""),
                row.get("local_resp", ""),
                safe_int(row.get("missed_bytes", "0")),
                row.get("history", ""),
                safe_int(row.get("orig_pkts", "0")),
                safe_int(row.get("orig_ip_bytes", "0")),
                safe_int(row.get("resp_pkts", "0")),
                safe_int(row.get("resp_ip_bytes", "0")),
                row.get("tunnel_parents", ""),
                row.get("ip_proto", ""),
            ])


if __name__ == "__main__":
    # Command (from root dir): uv run scripts/zeek_conn_to_csv.py data/DARPA_2000/inside true
    if len(sys.argv) < 3:
        print("Usage: python zeek_conn_to_csv.py <path_to_conn_logs_directory> <all_flows_flag>")
        sys.exit(1)

    logs_dir = Path(sys.argv[1]).resolve()

    all_flows_flag = sys.argv[2].lower() == "true"

    if all_flows_flag:
        print("Processing all flows into a single all_flows.csv...")
        conn_file = logs_dir / "conn.log"
        output_csv = conn_file.with_name(f"flows.csv")
        parse_conn_log(conn_file, output_csv)
        print(f"Saved all flows to: {output_csv}")
    else:
        print("Processing each phase conn.log file individually...")
        for conn_file in logs_dir.glob("phase*_conn.log"):
            phase_name = conn_file.stem.replace("_conn", "")
            output_csv = conn_file.with_name(f"{phase_name}_flows.csv")
            parse_conn_log(conn_file, output_csv)
            print(f"Processed {conn_file.name} → {output_csv.name}")
    print("Finished processing conn.log files.")