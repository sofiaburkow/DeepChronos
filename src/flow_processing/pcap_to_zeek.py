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


def main(dataset: str, scenario_network: str):
    raw_dir = Path(f"data/raw/{dataset}/{scenario_network}")
    zeek_dir = Path(f"data/interim/{dataset}/{scenario_network}/zeek_logs")

    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} does not exist")

    # ---- Process full PCAP ----
    full_pcaps = list(raw_dir.glob("LLS_DDOS_*.dump"))

    if full_pcaps:
        run_zeek(
            pcap_path=full_pcaps[0],
            output_dir=zeek_dir,
            output_name="all_conn.log",
        )
    else:
        print("[!] No full PCAP found")

    # ---- Process phase PCAPs ----
    phase_pcaps = sorted(raw_dir.glob("phase-*"))

    for pcap in phase_pcaps:
        # Extract phase number
        parts = pcap.name.split("-")
        phase_number = parts[1]

        run_zeek(
            pcap_path=pcap,
            output_dir=zeek_dir,
            output_name=f"phase{phase_number}_conn.log",
        )


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="darpa2000")
    ap.add_argument("--scenario_network", type=str, default="s1_inside")
    args = ap.parse_args()

    main(
        dataset=args.dataset, 
        scenario_network=args.scenario_network
    )