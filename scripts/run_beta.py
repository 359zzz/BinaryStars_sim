#!/usr/bin/env python
"""Scheme beta: run all PPO variants with progress tracking.

Usage:
    python scripts/run_beta.py
    python scripts/run_beta.py --max-parallel 8
    python scripts/run_beta.py --variants quantum_c quantum_decomp --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

VARIANTS = ["vanilla", "geometric", "coupling", "quantum_c", "quantum_decomp"]
SEEDS = list(range(10))

# CRITICAL: prevent BLAS thread deadlock in child processes
CHILD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}


def run_beta(
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    max_parallel: int = 6,
    config: str = "configs/beta.yaml",
) -> None:
    variants = variants or VARIANTS
    seeds = seeds or SEEDS

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.makedirs("results/beta", exist_ok=True)

    all_jobs = [(v, s) for v in variants for s in seeds]

    # Auto-resume: skip runs that already have history.json
    jobs = []
    skipped = 0
    for v, s in all_jobs:
        history_path = f"results/beta/{v}_seed{s}/history.json"
        if os.path.exists(history_path) and os.path.getsize(history_path) > 100:
            skipped += 1
        else:
            jobs.append((v, s))

    total = len(jobs)
    print(f"=== Scheme beta: {len(variants)} variants x {len(seeds)} seeds ===")
    if skipped:
        print(f"  Resuming: {skipped} already done, {total} remaining")
    print(f"  max_parallel={max_parallel}, BLAS threads=2")

    running: dict[tuple[str, int], subprocess.Popen] = {}
    completed = 0
    failed = 0
    t0 = time.time()
    job_idx = 0

    while job_idx < total or running:
        # Launch new jobs up to max_parallel
        while job_idx < total and len(running) < max_parallel:
            variant, seed = jobs[job_idx]
            log_path = f"results/beta/{variant}_seed{seed}.log"
            log_f = open(log_path, "w")
            proc = subprocess.Popen(
                [
                    sys.executable, "-u", "-m", "coupling_rl.train_ppo",
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
            running[(variant, seed)] = proc
            job_idx += 1

        # Check for completed
        done_keys = []
        for key, proc in running.items():
            ret = proc.poll()
            if ret is not None:
                proc._log_file.close()
                elapsed_job = time.time() - proc._start_time
                done_keys.append(key)
                if ret == 0:
                    completed += 1
                    status = "OK"
                else:
                    failed += 1
                    status = f"FAIL(rc={ret})"
                # Progress bar
                done_total = completed + failed
                elapsed = time.time() - t0
                if completed > 0:
                    eta = elapsed / done_total * (total - done_total)
                    eta_str = f"{eta/60:.0f}m"
                else:
                    eta_str = "?"
                bar_len = 30
                filled = int(bar_len * done_total / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"  [{bar}] {done_total}/{total} "
                    f"| {key[0]} s{key[1]} {status} ({elapsed_job:.0f}s) "
                    f"| elapsed {elapsed/60:.0f}m ETA {eta_str}"
                )

        for key in done_keys:
            del running[key]

        if running and not done_keys:
            time.sleep(2)

    elapsed = time.time() - t0
    print(f"\n=== Done: {completed} OK, {failed} failed, {elapsed/60:.0f} min total ===")

    if total == 0:
        print("All runs already complete!")

    if failed == 0:
        print("Generating figures...")
        subprocess.run(
            [sys.executable, "-m", "coupling_rl.plot_results", "--save_dir", "results/beta"],
            env=CHILD_ENV,
        )
        print("=== Scheme beta complete ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--config", default="configs/beta.yaml")
    args = parser.parse_args()
    run_beta(
        variants=args.variants,
        seeds=args.seeds,
        max_parallel=args.max_parallel,
        config=args.config,
    )


if __name__ == "__main__":
    main()
