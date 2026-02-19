import argparse
from pathlib import Path
import csv


def safe_int(val, default=0):
    """
    Safely convert a value to int, return default if conversion fails.
    """
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    """
    Safely convert a value to float, return default if conversion fails.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_conn_log(input_file: Path, output_csv: Path):
    """
    Convert a Zeek conn.log file into a structured CSV file.
    """

    print(f"[+] Converting {input_file.name} → {output_csv.name}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

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
                    fieldnames = line.split()[1:]
                num_header_lines += 1
                continue

            values = line.split("\t")
            row = dict(zip(fieldnames, values))

            ts = safe_float(row.get("ts", 0))
            dur = safe_float(row.get("duration", 0))

            writer.writerow([
                f"f{i - num_header_lines}",
                ts,
                ts + dur,
                dur,
                row.get("id.orig_h", ""),
                safe_int(row.get("id.orig_p", 0)),
                row.get("id.resp_h", ""),
                safe_int(row.get("id.resp_p", 0)),
                row.get("proto", ""),
                row.get("service", ""),
                safe_int(row.get("orig_bytes", 0)),
                safe_int(row.get("resp_bytes", 0)),
                row.get("conn_state", ""),
                row.get("local_orig", ""),
                row.get("local_resp", ""),
                safe_int(row.get("missed_bytes", 0)),
                row.get("history", ""),
                safe_int(row.get("orig_pkts", 0)),
                safe_int(row.get("orig_ip_bytes", 0)),
                safe_int(row.get("resp_pkts", 0)),
                safe_int(row.get("resp_ip_bytes", 0)),
                row.get("tunnel_parents", ""),
                row.get("ip_proto", ""),
            ])


def main(dataset: str, scenario_network: str, overwrite: bool):
    base_dir = Path("data/interim") / dataset / scenario_network

    zeek_dir = base_dir / "zeek_logs"
    flows_dir = base_dir / "flows_unlabeled"

    if not zeek_dir.exists():
        raise FileNotFoundError(f"{zeek_dir} does not exist")

    flows_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process full capture ----
    all_conn = zeek_dir / "all_conn.log"
    if all_conn.exists():
        output_csv = flows_dir / "all_flows.csv"
        if overwrite or not output_csv.exists():
            parse_conn_log(all_conn, output_csv)

    # ---- Process per-phase logs ----
    for conn_file in sorted(zeek_dir.glob("phase*_conn.log")):
        phase_name = conn_file.stem.replace("_conn", "")
        output_csv = flows_dir / f"{phase_name}_flows.csv"

        if overwrite or not output_csv.exists():
            parse_conn_log(conn_file, output_csv)

    print("[✓] Finished converting Zeek logs to CSV.")


if __name__ == "__main__":
    # Command: uv run python src/data/zeek_parser.py --dataset darpa2000 --scenario_network s1_inside --overwrite

    parser = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="darpa2000")
    ap.add_argument("--scenario_network", type=str, default="s1_inside")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        scenario_network=args.scenario_network,
        overwrite=args.overwrite
    )
