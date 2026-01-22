#!/usr/bin/env python3
"""Run multiple DPL experiments with different flag combinations.

This script invokes `src/DARPA/multi_step.py` repeatedly with combinations of
flags (function_name, pretrained, resampled, lookback_limit, seed) and saves
stdout/stderr to `src/DARPA/results/exp_logs/`.

Usage examples:
  uv run python src/DARPA/run_experiments.py --functions ddos,multi_step --seeds 123,456
  uv run python src/DARPA/run_experiments.py --dry-run

"""

from __future__ import annotations

import sys
import argparse
import itertools
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


LOG_DIR = Path("src/DARPA/results/exp_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def build_command(script: str, function_name: str, pretrained: bool, resampled: bool, lookback: int | None, seed: int) -> List[str]:
    cmd = [sys.executable, script, "--function_name", function_name, "--seed", str(seed)]
    if pretrained:
        cmd.append("--pretrained")
    if resampled:
        cmd.append("--resampled")
    # lookback is an optional integer; if None we omit the flag (full lookback)
    if lookback is not None:
        cmd.extend(["--lookback_limit", str(lookback)])
    return cmd


def run_cmd(cmd: List[str], log_path: Path) -> int:
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        out, _ = proc.communicate()
        log_path.write_text(out)
        return proc.returncode


def expand_bools(arg: str) -> List[bool]:
    """Parse a boolean flag argument. Accepts 'true','false','both'."""
    s = arg.lower()
    if s in ("both", "all"):
        return [False, True]
    if s in ("true", "1", "t"):
        return [True]
    if s in ("false", "0", "f"):
        return [False]
    raise ValueError(f"Cannot parse boolean choice: {arg}")


def parse_lookbacks(arg: str) -> List[int | None]:
    """Parse lookback limits: comma-separated ints or 'none'.

    Returns a list of Optional[int] where None means no limit (full lookback).
    Examples: 'none,20000' -> [None, 20000]
    """
    vals = [v.strip() for v in arg.split(",") if v.strip()]
    out: List[int | None] = []
    for v in vals:
        if v.lower() in ("none", "null", ""):
            out.append(None)
            continue
        out.append(int(v))
    return out


def main():
    ap = argparse.ArgumentParser(description="Run many multi_step experiments")
    ap.add_argument("--script", default="src/DARPA/multi_step.py", help="Path to multi_step script")
    ap.add_argument("--functions", default="ddos,multi_step", help="Comma-separated function names")
    ap.add_argument("--pretrained", default="both", help="true|false|both")
    ap.add_argument("--resampled", default="both", help="true|false|both")
    ap.add_argument("--lookbacks", default="none,20000", help="Comma-separated lookback limits, use 'none' for full lookback. Example: 'none,20000'")
    ap.add_argument("--seeds", default="456", help="Comma-separated seeds")
    ap.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    ap.add_argument("--dry-run", action="store_true", default=False, help="Print commands but do not run")
    args = ap.parse_args()

    functions = [f.strip() for f in args.functions.split(",") if f.strip()]
    pretrained_choices = expand_bools(args.pretrained)
    resampled_choices = expand_bools(args.resampled)
    lookback_choices = parse_lookbacks(args.lookbacks)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    combos = list(itertools.product(functions, pretrained_choices, resampled_choices, lookback_choices, seeds))
    print(f"Will run {len(combos)} experiments (parallel={args.parallel})")

    if args.dry_run:
        for fn, pre, res, lb, sd in combos:
            print(" ", " ".join(build_command(args.script, fn, pre, res, lb, sd)))
        return

    # run sequentially or in parallel
    results = []
    if args.parallel <= 1:
        for fn, pre, res, lb, sd in combos:
            lb_str = "full" if lb is None else str(lb)
            name = f"{fn}_pre{int(pre)}_res{int(res)}_lb{lb_str}_s{sd}"
            logfile = LOG_DIR / f"{name}.log"
            cmd = build_command(args.script, fn, pre, res, lb, sd)
            print("RUN:", " ".join(cmd), "->", logfile)
            rc = run_cmd(cmd, logfile)
            print(name, "done rc=", rc)
            results.append((name, rc, logfile))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            future_map = {}
            for fn, pre, res, lb, sd in combos:
                lb_str = "full" if lb is None else str(lb)
                name = f"{fn}_pre{int(pre)}_res{int(res)}_lb{lb_str}_s{sd}"
                logfile = LOG_DIR / f"{name}.log"
                cmd = build_command(args.script, fn, pre, res, lb, sd)
                print("SCHEDULE:", " ".join(cmd), "->", logfile)
                fut = ex.submit(run_cmd, cmd, logfile)
                future_map[fut] = (name, logfile)

            for fut in as_completed(future_map):
                name, logfile = future_map[fut]
                try:
                    rc = fut.result()
                except Exception as e:
                    rc = -1
                    logfile.write_text(str(e))
                print(name, "done rc=", rc, "log=", logfile)
                results.append((name, rc, logfile))

    # summary
    print("\nSummary:")
    for name, rc, logfile in results:
        print(f"{name}: rc={rc} log={logfile}")


if __name__ == "__main__":
    main()
