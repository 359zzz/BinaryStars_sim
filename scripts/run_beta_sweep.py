#!/usr/bin/env python
"""Lambda sweep for Scheme beta: find optimal coupling_lambda.

Runs quantum_c + vanilla at multiple lambda values to find the sweet spot
where coupling reward is meaningful but doesn't overwhelm the task.

Usage:
    python scripts/run_beta_sweep.py
    python scripts/run_beta_sweep.py --lambdas 0.5 1.0 2.0 5.0 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
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
}

SWEEP_BASE = "results/beta_sweep"


def run_sweep(
    lambdas: list[float] | None = None,
    seeds: list[int] | None = None,
    max_parallel: int = 6,
    base_config: str = "configs/beta_v2.yaml",
) -> None:
    lambdas = lambdas or [0.5, 1.0, 2.0, 5.0]
    seeds = seeds or [0, 1]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    with open(base_config) as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(SWEEP_BASE, exist_ok=True)

    # Build all jobs.
    # train_ppo.py saves to {save_dir}/{variant}_seed{seed}/
    # So we set save_dir = SWEEP_BASE/lam{X} for each lambda.
    # Vanilla baseline uses save_dir = SWEEP_BASE/vanilla
    jobs = []  # (variant, lambda, seed, save_dir, history_path)

    # Vanilla baseline (only need one set, lambda=0 doesn't matter)
    v_dir = f"{SWEEP_BASE}/vanilla"
    for seed in seeds:
        hp = f"{v_dir}/vanilla_seed{seed}/history.json"
        if not (os.path.exists(hp) and os.path.getsize(hp) > 100):
            jobs.append(("vanilla", 0.0, seed, v_dir))

    # quantum_c per lambda
    for lam in lambdas:
        lam_dir = f"{SWEEP_BASE}/lam{lam}"
        for seed in seeds:
            hp = f"{lam_dir}/quantum_c_seed{seed}/history.json"
            if not (os.path.exists(hp) and os.path.getsize(hp) > 100):
                jobs.append(("quantum_c", lam, seed, lam_dir))

    total = len(jobs)
    if total == 0:
        print("All sweep runs already complete!")
        _analyze_sweep(lambdas, seeds)
        return

    print(f"=== Lambda Sweep: {len(lambdas)} lambdas x {len(seeds)} seeds ===")
    print(f"  lambdas={lambdas}, {total} runs, max_parallel={max_parallel}")

    running = {}
    completed = 0
    failed = 0
    t0 = time.time()
    job_idx = 0
    tmp_configs = []

    while job_idx < total or running:
        while job_idx < total and len(running) < max_parallel:
            variant, lam, seed, save_dir = jobs[job_idx]
            os.makedirs(save_dir, exist_ok=True)

            # Create per-job config
            cfg = dict(base_cfg)
            cfg["coupling_lambda"] = lam
            cfg["save_dir"] = save_dir

            job_key = f"{variant}_lam{lam}_s{seed}"
            tmp_cfg = f"{SWEEP_BASE}/_cfg_{job_key}.yaml"
            with open(tmp_cfg, "w") as f:
                yaml.dump(cfg, f)
            tmp_configs.append(tmp_cfg)

            log_path = f"{save_dir}/{variant}_seed{seed}.log"
            log_f = open(log_path, "w")
            proc = subprocess.Popen(
                [
                    sys.executable, "-u", "-m", "coupling_rl.train_ppo",
                    "--config", tmp_cfg,
                    "--variant", variant,
                    "--seed", str(seed),
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=CHILD_ENV,
            )
            proc._log_file = log_f
            proc._start_time = time.time()
            running[job_key] = proc
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
                print(
                    f"  [{done_total}/{total}] {key} {status} ({dt:.0f}s) "
                    f"| elapsed {elapsed/60:.0f}m ETA {eta/60:.0f}m"
                )

        for key in done_keys:
            del running[key]

        if running and not done_keys:
            time.sleep(2)

    for f in tmp_configs:
        try:
            os.remove(f)
        except OSError:
            pass

    elapsed = time.time() - t0
    print(f"\n=== Sweep done: {completed} OK, {failed} failed, {elapsed/60:.0f}m ===")
    _analyze_sweep(lambdas, seeds)


def _analyze_sweep(lambdas: list[float], seeds: list[int]) -> None:
    """Quick analysis of sweep results."""
    import numpy as np

    print("\n" + "=" * 60)
    print("  LAMBDA SWEEP RESULTS")
    print("=" * 60)
    print(f"  {'Lambda':>8s} {'Reward':>14s} {'Success':>14s} {'vs Vanilla':>12s}")
    print("-" * 60)

    # Vanilla baseline
    v_rewards, v_success = [], []
    for seed in seeds:
        hp = f"{SWEEP_BASE}/vanilla/vanilla_seed{seed}/history.json"
        if os.path.exists(hp):
            h = json.load(open(hp))
            last = h[-20:] if len(h) >= 20 else h
            v_rewards.append(float(np.mean([e["mean_reward"] for e in last])))
            v_success.append(float(np.mean([e["success_rate"] for e in last])))
    v_mean = float(np.mean(v_rewards)) if v_rewards else 0.0
    v_std = float(np.std(v_rewards)) if v_rewards else 0.0
    vs_mean = float(np.mean(v_success)) if v_success else 0.0
    vs_std = float(np.std(v_success)) if v_success else 0.0
    print(
        f"  {'vanilla':>8s} {v_mean:7.0f}+/-{v_std:<4.0f}  "
        f"{vs_mean:.4f}+/-{vs_std:.4f}  {'baseline':>12s}"
    )

    best_lam = None
    best_score = -float("inf")
    for lam in lambdas:
        rewards, successes = [], []
        for seed in seeds:
            hp = f"{SWEEP_BASE}/lam{lam}/quantum_c_seed{seed}/history.json"
            if os.path.exists(hp):
                h = json.load(open(hp))
                last = h[-20:] if len(h) >= 20 else h
                rewards.append(float(np.mean([e["mean_reward"] for e in last])))
                successes.append(float(np.mean([e["success_rate"] for e in last])))
        if not rewards:
            print(f"  {lam:8.1f}  {'(no data)':>14s}")
            continue
        r_mean = float(np.mean(rewards))
        r_std = float(np.std(rewards))
        s_mean = float(np.mean(successes))
        s_std = float(np.std(successes))
        # Score: prefer higher success, then higher reward
        score = s_mean * 1000 + r_mean * 0.001
        marker = ""
        if score > best_score:
            best_score = score
            best_lam = lam
            marker = " <--"
        print(
            f"  {lam:8.1f} {r_mean:7.0f}+/-{r_std:<4.0f}  "
            f"{s_mean:.4f}+/-{s_std:.4f}  "
            f"{r_mean - v_mean:+8.0f}{marker}"
        )

    print("-" * 60)
    if best_lam is not None:
        print(f"  Recommended: coupling_lambda = {best_lam}")
        print(f"\n  Next steps:")
        print(f"  1. Update configs/beta_v2.yaml: coupling_lambda: {best_lam}")
        print(f"  2. rm -rf results/beta/")
        print(f"  3. python scripts/run_beta.py --config configs/beta_v2.yaml --max-parallel 6")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Lambda sweep for Scheme beta")
    parser.add_argument("--lambdas", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--config", default="configs/beta_v2.yaml")
    args = parser.parse_args()
    run_sweep(
        lambdas=args.lambdas,
        seeds=args.seeds,
        max_parallel=args.max_parallel,
        base_config=args.config,
    )


if __name__ == "__main__":
    main()
