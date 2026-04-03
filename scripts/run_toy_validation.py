#!/usr/bin/env python
"""Run toy dual-arm modal action validation experiment.

Usage:
    python -m scripts.run_toy_validation                               # all (3 variants x 5 seeds)
    python -m scripts.run_toy_validation --variant modal_action --seed 0  # single run
    python -m scripts.run_toy_validation --analyze-only                # analyze existing results
    python -m scripts.run_toy_validation --mass-sweep                  # sweep object masses
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import time

import torch
import yaml


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "toy.yaml",
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    config_path: str | None = None,
    device: str | None = None,
    mass_sweep: bool = False,
) -> None:
    from coupling_rl.train_toy import train_toy

    cfg = load_config(config_path)
    variants = variants or cfg.get("variants", ["vanilla", "coupling_features", "modal_action"])
    seeds = seeds or cfg.get("seeds", [0, 1, 2, 3, 4])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg.get("save_dir", "results/toy")

    if mass_sweep:
        masses = cfg.get("mass_sweep", [0.0, 0.1, 0.3, 0.5, 1.0])
    else:
        masses = [cfg.get("object_mass", 0.5)]

    total_runs = len(masses) * len(variants) * len(seeds)
    run_idx = 0

    for mass in masses:
        mass_cfg = dict(cfg)
        mass_cfg["object_mass"] = mass
        mass_dir = save_dir if len(masses) == 1 else os.path.join(save_dir, f"mass_{mass:.1f}")
        mass_cfg["save_dir"] = mass_dir

        for variant in variants:
            for seed in seeds:
                run_idx += 1
                out_dir = os.path.join(mass_dir, f"{variant}_seed{seed}")
                hist_path = os.path.join(out_dir, "history.json")

                if os.path.exists(hist_path):
                    print(f"[{run_idx}/{total_runs}] SKIP {variant}/seed{seed} (exists)")
                    continue

                print(f"\n[{run_idx}/{total_runs}] {variant}/seed{seed}"
                      f" (mass={mass:.1f}, device={device})")
                t0 = time.time()
                train_toy(
                    variant=variant,
                    seed=seed,
                    config=mass_cfg,
                    device=device,
                    save_dir=mass_dir,
                )
                elapsed = time.time() - t0
                print(f"  Completed in {elapsed:.0f}s")

    print(f"\nAll {total_runs} runs done. Results in {save_dir}/")


def analyze(config_path: str | None = None) -> None:
    """Run analysis script on existing results."""
    from scripts.analyze_toy import main as analyze_main
    cfg = load_config(config_path)
    analyze_main(save_dir=cfg.get("save_dir", "results/toy"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy dual-arm validation")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--mass-sweep", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze(args.config)
        return

    variants = [args.variant] if args.variant else None
    seeds = [args.seed] if args.seed is not None else None

    run_experiment(
        variants=variants,
        seeds=seeds,
        config_path=args.config,
        device=args.device,
        mass_sweep=args.mass_sweep,
    )

    # Auto-analyze after training
    print("\n--- Analysis ---")
    analyze(args.config)


if __name__ == "__main__":
    main()
