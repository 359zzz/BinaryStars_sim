#!/usr/bin/env python
"""Run dual-arm transport experiments: Scheme A (weld) + Scheme B (virtual) in parallel.

Usage:
    python scripts/run_dualarm_v2.py                    # both schemes
    python scripts/run_dualarm_v2.py --schemes weld     # only weld
    python scripts/run_dualarm_v2.py --schemes virtual  # only virtual
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

CHILD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}

SCHEME_CONFIGS = {
    "weld": "configs/dualarm_weld.yaml",
    "virtual": "configs/dualarm_virtual.yaml",
}

VARIANTS = ["vanilla", "coupling", "quantum_c", "quantum_decomp"]
SEEDS = list(range(10))


def run_all(
    schemes: list[str] | None = None,
    max_parallel: int = 6,
) -> None:
    schemes = schemes or ["weld", "virtual"]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Build job list across both schemes
    jobs = []  # (scheme, variant, seed, config, save_dir)
    skipped = 0
    for scheme in schemes:
        config = SCHEME_CONFIGS[scheme]
        save_dir = f"results/dualarm_{scheme}"
        os.makedirs(save_dir, exist_ok=True)
        for v in VARIANTS:
            for s in SEEDS:
                hp = f"{save_dir}/{v}_seed{s}/history.json"
                if os.path.exists(hp) and os.path.getsize(hp) > 100:
                    skipped += 1
                else:
                    jobs.append((scheme, v, s, config, save_dir))

    total = len(jobs)
    print(f"=== Dual-Arm Transport: schemes={schemes} ===")
    print(f"  {len(VARIANTS)} variants x {len(SEEDS)} seeds x {len(schemes)} schemes = {total + skipped} total")
    if skipped:
        print(f"  Resuming: {skipped} done, {total} remaining")
    print(f"  max_parallel={max_parallel}")

    if total == 0:
        print("All runs complete!")
        return

    running = {}
    completed = 0
    failed = 0
    t0 = time.time()
    job_idx = 0

    while job_idx < total or running:
        while job_idx < total and len(running) < max_parallel:
            scheme, variant, seed, config, save_dir = jobs[job_idx]

            log_path = f"{save_dir}/{variant}_seed{seed}.log"
            log_f = open(log_path, "w")
            proc = subprocess.Popen(
                [
                    sys.executable, "-u", "-m", "coupling_rl.train_dualarm",
                    "--config", config,
                    "--variant", variant,
                    "--seed", str(seed),
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=CHILD_ENV,
            )
            proc._log_file = log_f
            proc._start_time = time.time()
            key = f"{scheme}/{variant}_s{seed}"
            running[key] = proc
            job_idx += 1

        done_keys = []
        for key, proc in running.items():
            ret = proc.poll()
            if ret is not None:
                proc._log_file.close()
                dt = time.time() - proc._start_time
                done_keys.append(key)
                if ret == 0:
                    completed += 1
                    status = "OK"
                else:
                    failed += 1
                    status = f"FAIL(rc={ret})"
                done_total = completed + failed
                elapsed = time.time() - t0
                eta = elapsed / done_total * (total - done_total) if done_total > 0 else 0
                bar_len = 30
                filled = int(bar_len * done_total / total)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                print(
                    f"  [{bar}] {done_total}/{total} "
                    f"| {key} {status} ({dt:.0f}s) "
                    f"| elapsed {elapsed/60:.0f}m ETA {eta/60:.0f}m"
                )

        for key in done_keys:
            del running[key]

        if running and not done_keys:
            time.sleep(2)

    elapsed = time.time() - t0
    print(f"\n=== Done: {completed} OK, {failed} failed, {elapsed/60:.0f}m total ===")

    # Quick summary
    if failed == 0:
        _print_summary(schemes)


def _print_summary(schemes: list[str]) -> None:
    """Print quick comparison of results."""
    import json
    import numpy as np

    for scheme in schemes:
        save_dir = f"results/dualarm_{scheme}"
        print(f"\n=== Scheme {scheme.upper()} ===")
        print(f"  {'Variant':20s} {'Reward':>14s} {'Success':>14s}")
        print("  " + "-" * 50)
        for v in VARIANTS:
            rewards, success = [], []
            for s in SEEDS:
                hp = f"{save_dir}/{v}_seed{s}/history.json"
                if os.path.exists(hp):
                    h = json.load(open(hp))
                    last20 = h[-20:] if len(h) >= 20 else h
                    rewards.append(np.mean([e["mean_reward"] for e in last20]))
                    success.append(np.mean([e["success_rate"] for e in last20]))
            if rewards:
                print(
                    f"  {v:20s} {np.mean(rewards):7.0f}+/-{np.std(rewards):<5.0f} "
                    f"{np.mean(success):.4f}+/-{np.std(success):.4f}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", default=None,
                        choices=["weld", "virtual"])
    parser.add_argument("--max-parallel", type=int, default=6)
    args = parser.parse_args()
    run_all(schemes=args.schemes, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
