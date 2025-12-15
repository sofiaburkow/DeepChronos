import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

def sanitize_name(cmd: str) -> str:
    '''
    Create a sanitized name for a command, suitable for filenames.
    '''
    parts = shlex.split(cmd)

    model_token = next((p for p in parts if p.endswith(".py")), None)
    model_name = Path(model_token).stem if model_token else "model"

    dataset_token = parts[-2] 
    print(dataset_token)
    p = Path(dataset_token.rstrip("/"))
    dataset_hint = "-".join(p.parts[-3:])

    return model_name, f"{model_name}__{dataset_hint}"


def run_command(cmd: str, log_dir: Path) -> dict:
    """Run a shell command and capture logs. Returns dict with status info."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    model_name, file_name = sanitize_name(cmd)
    model_log_dir = log_dir / model_name
    model_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = model_log_dir / f"{file_name}.log"

    with open(log_file, "wb") as f:
        f.write(f"COMMAND: {cmd}\nSTART: {ts}\n\n".encode())
        f.flush()
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            f.write(line)
            f.flush()
        ret = proc.wait()
        end_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        f.write(f"\nEND: {end_ts}\nRETURN_CODE: {ret}\n".encode())

    return {
        "cmd": cmd,
        "log": str(log_file),
        "returncode": ret,
    }


def build_manifest_from_lists(models_file: str, datasets_file: str, cmd_template: str = "uv run python {model} {dataset}"):
    """Build a manifest (list of shell commands) by combining models and datasets.

    - models_file: path to JSON list of model script paths (e.g. ["experiments/models/mlp.py", ...])
    - datasets_file: path to JSON list of dataset directory paths
    - cmd_template: python format string with {model} and {dataset}
    """
    with open(models_file) as f:
        models = json.load(f)
    with open(datasets_file) as f:
        datasets = json.load(f)
    if not isinstance(models, list) or not isinstance(datasets, list):
        raise ValueError("models_file and datasets_file must be JSON lists")
    manifest = []
    for m in models:
        model_name = Path(m).stem
        print(f"Model: {model_name}")
        for d in datasets:
            manifest.append(cmd_template.format(model=m, dataset=d))
    print()

    return manifest


def main():
    ap = argparse.ArgumentParser(description="Run a list of training commands and collect logs")
    ap.add_argument("--models-file", default="experiments/models_list.json", help="Path to JSON manifest (list of model script paths)")
    ap.add_argument("--datasets-file", default="experiments/datasets_list.json", help="Path to JSON manifest (list of dataset directory paths)")
    ap.add_argument("--class-weights", type=bool, default=False, help="Append a class-weights flag to each training command")
    ap.add_argument("--parallel", type=int, default=1, help="Number of parallel jobs (default 1)")
    ap.add_argument("--log-dir", default="experiments/logs", help="Directory to store logs")
    ap.add_argument("--continue-on-error", action="store_true", help="Keep running remaining jobs when one fails")
    args = ap.parse_args()

    manifest = build_manifest_from_lists(args.models_file, args.datasets_file)
    
    if len(manifest) == 0:
        print("Manifest is empty. Edit DEFAULT_MANIFEST in this script or pass --manifest.")
        sys.exit(1)

    manifest = [f"{cmd} {args.class_weights}" for cmd in manifest]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if args.parallel <= 1:
        for cmd in manifest:
            print(f"Running: {cmd}")
            res = run_command(cmd, log_dir)
            results.append(res)
            print(f" -> returned {res['returncode']}, log: {res['log']}")
            if res["returncode"] != 0 and not args.continue_on_error:
                print("Stopping due to non-zero return code. Use --continue-on-error to continue.")
                break
    else:
        # parallel
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            future_to_cmd = {ex.submit(run_command, cmd, log_dir): cmd for cmd in manifest}
            for fut in as_completed(future_to_cmd):
                cmd = future_to_cmd[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"cmd": cmd, "log": None, "returncode": -1, "error": str(e)}
                results.append(res)
                print(f"Finished: {cmd} -> {res.get('returncode')} (log: {res.get('log')})")
                if res.get("returncode", -1) != 0 and not args.continue_on_error:
                    print("A job failed; other running jobs will continue but no new jobs will be submitted.")

    # summary
    print("\n=== Summary ===")
    ok = [r for r in results if r.get('returncode', 1) == 0]
    bad = [r for r in results if r.get('returncode', 1) != 0]
    print(f"Total jobs: {len(results)}; successful: {len(ok)}; failed: {len(bad)}")
    if bad:
        print("Failed jobs (cmd -> log -> rc):")
        for r in bad:
            print(r.get('cmd'), "->", r.get('log'), "->", r.get('returncode'))


if __name__ == '__main__':
    # Command: uv run python experiments/train_all.py --models-file experiments/models_list.json --datasets-file experiments/datasets_list.json --parallel 5
    main()