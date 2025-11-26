import sys
from pathlib import Path
import csv

def safe_float(val):
    '''
    Safely convert a value to float, return 0.0 if conversion fails.
    '''
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

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
            "flow_id","ts_start","ts_end","duration",
            "src_ip","src_port","dst_ip","dst_port",
            "proto","service","orig_bytes","resp_bytes",
            "orig_pkts","resp_pkts",
            # "conn_state"
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

            src_ip = row.get("id.orig_h", "")
            src_port = row.get("id.orig_p", "")
            dst_ip = row.get("id.resp_h", "")
            dst_port = row.get("id.resp_p", "")

            proto = row.get("proto", "")    
            service = row.get("service", "")
            orig_bytes = row.get("orig_bytes", 0)
            resp_bytes = row.get("resp_bytes", 0)
            orig_pkts = row.get("orig_pkts", 0)
            resp_pkts = row.get("resp_pkts", 0)
            # conn_state = row.get("conn_state", "")

            writer.writerow([
                flow_id,
                ts,
                ts + dur,
                dur,
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                proto,
                service,
                orig_bytes,
                resp_bytes,
                orig_pkts,
                resp_pkts,
                # conn_state,
            ])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zeek_conn_to_csv.py <path_to_input_conn.log>")
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    dataset_name = input_path.parent.name # either inside or dmz
    output_path = input_path.with_name(f"{dataset_name}_flows.csv")

    parse_conn_log(input_path, output_path)