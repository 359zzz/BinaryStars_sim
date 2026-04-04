#!/usr/bin/env python
"""Run dual-arm quantum vs classical RL experiment.

Usage:
    python scripts/run_dualarm.py
    python scripts/run_dualarm.py --max-parallel 6
    python scripts/run_dualarm.py --variants quantum_c --seeds 0 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import yaml

CHILD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}


def run_dualarm(
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    max_parallel: int = 6,
    config: str = "configs/dualarm.yaml",
) -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    variants = variants or cfg.get("variants", ["vanilla", "coupling", "quantum_c", "quantum_decomp"])
    seeds = seeds or cfg.get("seeds", list(range(10)))
    save_dir = cfg.get("save_dir", "results/dualarm")
    os.makedirs(save_dir, exist_ok=True)

    all_jobs = [(v, s) for v in variants for s in seeds]

    # Auto-resume
    jobs = []
    skipped = 0
    for v, s in all_jobs:
        hp = f"{save_dir}/{v}_seed{s}/history.json"
        if os.path.exists(hp) and os.path.getsize(hp) > 100:
            skipped += 1
        else:
            jobs.append((v, s))

    total = len(jobs)
    print(f"=== Dual-Arm RL: {len(variants)} variants x {len(seeds)} seeds ===")
    if skipped:
        print(f"  Resuming: {skipped} done, {total} remaining")
    print(f"  max_parallel={max_parallel}, BLAS threads=2")

    if total == 0:
        print("All runs already complete!")
        return

    running = {}
    completed = 0
    failed = 0
    t0 = time.time()
    job_idx = 0

    while job_idx < total or running:
        while job_idx < total and len(running) < max_parallel:
            variant, seed = jobs[job_idx]
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
            running[(variant, seed)] = proc
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
                    f"| {key[0]} s{key[1]} {status} ({dt:.0f}s) "
                    f"| elapsed {elapsed/60:.0f}m ETA {eta/60:.0f}m"
                )

        for key in done_keys:
            del running[key]

        if running and not done_keys:
            time.sleep(2)

    elapsed = time.time() - t0
    print(f"\n=== Done: {completed} OK, {failed} failed, {elapsed/60:.0f}m total ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--config", default="configs/dualarm.yaml")
    args = parser.parse_args()
    run_dualarm(
        variants=args.variants,
        seeds=args.seeds,
        max_parallel=args.max_parallel,
        config=args.config,
    )


if __name__ == "__main__":
    main()
