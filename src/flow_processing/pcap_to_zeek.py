import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path


def run_zeek(pcap_path: Path, output_dir: Path, output_name: str):
    """
    Run Zeek on a PCAP file and store only conn.log
    as <output_name> in output_dir.
    """

    pcap_path = pcap_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Processing {pcap_path.name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        subprocess.run(
            ["zeek", "-Cr", str(pcap_path)],
            cwd=tmpdir_path,
            check=True,
        )

        tmp_conn = tmpdir_path / "conn.log"

        if not tmp_conn.exists():
            raise FileNotFoundError(f"conn.log not generated for {pcap_path.name}")

        shutil.move(tmp_conn, output_dir / output_name)


def process_darpa(raw_dir, zeek_dir):
    full_pcaps = list(raw_dir.glob("LLS_DDOS_*.dump"))
    if full_pcaps:
        run_zeek(
            pcap_path=full_pcaps[0],
            output_dir=zeek_dir,
            output_name="all_conn.log",
        )
    else:
        print("[!] No full PCAP found")

    phase_pcaps = sorted(raw_dir.glob("phase-*"))
    for pcap in phase_pcaps:
        parts = pcap.name.split("-")
        phase_number = parts[1]
        run_zeek(
            pcap_path=pcap,
            output_dir=zeek_dir,
            output_name=f"phase{phase_number}_conn.log",
        )


def process_ait(raw_dir, zeek_dir):
    pcaps = list(raw_dir.glob("log_*.pcap"))
    for pcap in pcaps:
        run_zeek(
            pcap_path=pcap,
            output_dir=zeek_dir,
            output_name=f"{pcap.stem}_conn.log",
        )


def main(dataset: str, scenario: str):
    zeek_dir = Path(f"data/interim/{dataset}/{scenario}/zeek_logs")

    if dataset == "darpa2000":
        raw_dir = Path(f"data/raw/{dataset}/{scenario}")
        if not raw_dir.exists():
            raise FileNotFoundError(f"{raw_dir} does not exist")
        
        process_darpa(raw_dir, zeek_dir)

    elif dataset == "aitv2":
        raw_dir = Path(f"data/raw/{dataset}/{scenario}/pcaps")
        if not raw_dir.exists():
            raise FileNotFoundError(f"{raw_dir} does not exist")
        
        process_ait(raw_dir, zeek_dir)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    # uv run python -m src.flow_processing.pcap_to_zeek --dataset aitv2 --scenario fox

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="darpa2000")
    parser.add_argument("--scenario", type=str, default="s1_inside")
    args = parser.parse_args()

    main(
        dataset=args.dataset, 
        scenario=args.scenario
    )