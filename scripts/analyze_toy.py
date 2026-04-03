#!/usr/bin/env python
"""Analyze toy dual-arm validation results and generate figures.

Output:
    figures/toy_learning_curves.pdf    — RMSE vs steps (3 variants, shaded +/- std)
    figures/toy_sample_efficiency.pdf  — steps to threshold bar chart
    figures/toy_final_comparison.pdf   — final RMSE bar chart
    figures/toy_mass_sweep.pdf         — speedup ratio vs object mass (if mass sweep data)
    Terminal: summary table

Usage:
    python -m scripts.analyze_toy
    python -m scripts.analyze_toy --save-dir results/toy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANT_STYLES = {
    "vanilla":            {"color": "#4C72B0", "label": "Vanilla PPO"},
    "coupling_features":  {"color": "#DD8452", "label": "Coupling Features"},
    "modal_action":       {"color": "#55A868", "label": "Modal Action (ours)"},
}
FIG_DIR = "figures"
RMSE_THRESHOLD = 0.25  # rad — "converged" threshold for sample efficiency


def smooth(data: np.ndarray, window: int = 5) -> np.ndarray:
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_histories(save_dir: str) -> dict[str, list[list[dict]]]:
    """Load training histories grouped by variant."""
    results: dict[str, list[list[dict]]] = {}
    base = Path(save_dir)
    if not base.exists():
        return results
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # Handle mass_X.X subdirs
        if name.startswith("mass_"):
            continue
        variant = name.rsplit("_seed", 1)[0]
        hist_path = d / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                results.setdefault(variant, []).append(json.load(f))
    return results


def _extract_metric(histories: dict, metric: str, window: int = 5):
    """Extract smoothed metric arrays aligned by steps."""
    out = {}
    for variant, runs in histories.items():
        arrays = []
        for run in runs:
            vals = [e.get(metric, 0.0) for e in run]
            arrays.append(vals)
        min_len = min(len(a) for a in arrays)
        arr = np.array([a[:min_len] for a in arrays])
        steps = np.array([e["step"] for e in runs[0][:min_len]])
        out[variant] = (steps, arr)
    return out


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

def plot_learning_curves(histories: dict, fig_dir: str = FIG_DIR) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    data = _extract_metric(histories, "mean_rmse")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ["vanilla", "coupling_features", "modal_action"]:
        if variant not in data:
            continue
        steps, arr = data[variant]
        style = VARIANT_STYLES.get(variant, {"color": "gray", "label": variant})
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        mean_s = smooth(mean)
        std_s = smooth(std)
        steps_s = steps[:len(mean_s)]
        ax.plot(steps_s, mean_s, color=style["color"], label=style["label"], linewidth=2)
        ax.fill_between(steps_s, mean_s - std_s, mean_s + std_s,
                         alpha=0.15, color=style["color"])

    ax.axhline(RMSE_THRESHOLD, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({RMSE_THRESHOLD})")
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Tracking RMSE (rad)", fontsize=12)
    ax.set_title("Toy Dual-Arm: Learning Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "toy_learning_curves.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_sample_efficiency(histories: dict, fig_dir: str = FIG_DIR) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    data = _extract_metric(histories, "mean_rmse")
    if not data:
        return

    variant_order = ["vanilla", "coupling_features", "modal_action"]
    labels = []
    means = []
    stds = []
    colors = []

    for variant in variant_order:
        if variant not in data:
            continue
        steps, arr = data[variant]
        # Steps to reach threshold for each seed
        steps_to_threshold = []
        for seed_arr in arr:
            below = np.where(seed_arr < RMSE_THRESHOLD)[0]
            if len(below) > 0:
                steps_to_threshold.append(steps[below[0]])
            else:
                steps_to_threshold.append(steps[-1])
        style = VARIANT_STYLES.get(variant, {"color": "gray", "label": variant})
        labels.append(style["label"])
        means.append(np.mean(steps_to_threshold))
        stds.append(np.std(steps_to_threshold))
        colors.append(style["color"])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Steps to RMSE < {:.2f}".format(RMSE_THRESHOLD), fontsize=12)
    ax.set_title("Sample Efficiency", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(fig_dir, "toy_sample_efficiency.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_final_comparison(histories: dict, fig_dir: str = FIG_DIR) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    data = _extract_metric(histories, "mean_rmse")
    if not data:
        return

    variant_order = ["vanilla", "coupling_features", "modal_action"]
    labels, final_means, final_stds, colors = [], [], [], []

    for variant in variant_order:
        if variant not in data:
            continue
        _, arr = data[variant]
        # Last 10% of training
        n_tail = max(arr.shape[1] // 10, 1)
        tail = arr[:, -n_tail:]
        per_seed = tail.mean(axis=1)
        style = VARIANT_STYLES.get(variant, {"color": "gray", "label": variant})
        labels.append(style["label"])
        final_means.append(per_seed.mean())
        final_stds.append(per_seed.std())
        colors.append(style["color"])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    ax.bar(x, final_means, yerr=final_stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Final Tracking RMSE (rad)", fontsize=12)
    ax.set_title("Final Performance Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(fig_dir, "toy_final_comparison.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_mass_sweep(save_dir: str, fig_dir: str = FIG_DIR) -> None:
    """Plot speedup ratio vs object mass (requires mass sweep data)."""
    base = Path(save_dir)
    mass_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("mass_")])
    if not mass_dirs:
        return

    os.makedirs(fig_dir, exist_ok=True)
    masses = []
    speedup_means = []
    speedup_stds = []

    for md in mass_dirs:
        mass = float(md.name.split("_")[1])
        histories = load_histories(str(md))
        if "vanilla" not in histories or "modal_action" not in histories:
            continue

        data = _extract_metric(histories, "mean_rmse")
        if "vanilla" not in data or "modal_action" not in data:
            continue

        # Compute speedup = vanilla_steps / modal_steps
        ratios = []
        for variant in ["vanilla", "modal_action"]:
            steps, arr = data[variant]
            s_list = []
            for seed_arr in arr:
                below = np.where(seed_arr < RMSE_THRESHOLD)[0]
                s_list.append(steps[below[0]] if len(below) > 0 else steps[-1])
            ratios.append(np.array(s_list))

        if len(ratios[0]) > 0 and len(ratios[1]) > 0:
            speedup = ratios[0].mean() / max(ratios[1].mean(), 1)
            masses.append(mass)
            speedup_means.append(speedup)
            # Approximate std via delta method
            speedup_stds.append(speedup * np.sqrt(
                (ratios[0].std() / max(ratios[0].mean(), 1))**2 +
                (ratios[1].std() / max(ratios[1].mean(), 1))**2
            ))

    if not masses:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(masses, speedup_means, yerr=speedup_stds, marker="o",
                capsize=5, color="#55A868", linewidth=2, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Object Mass (kg)", fontsize=12)
    ax.set_ylabel("Speedup (vanilla / modal)", fontsize=12)
    ax.set_title("Modal Action Speedup vs Object Mass", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "toy_mass_sweep.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_capacity_sweep(save_dir: str, fig_dir: str = FIG_DIR) -> None:
    """Plot final RMSE vs hidden_dim for vanilla vs modal_action.

    Expects results in save_dir/h32/, save_dir/h64/, save_dir/h128/.
    """
    base = Path(save_dir)
    hidden_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("h")],
        key=lambda d: int(d.name[1:]),
    )
    if not hidden_dirs:
        return

    os.makedirs(fig_dir, exist_ok=True)
    hidden_vals = []
    results: dict[str, tuple[list, list]] = {}  # variant -> (means, stds)

    for hd in hidden_dirs:
        h = int(hd.name[1:])
        hidden_vals.append(h)
        histories = load_histories(str(hd))
        data = _extract_metric(histories, "mean_rmse")

        for variant in ["vanilla", "coupling_features", "modal_action"]:
            if variant not in data:
                continue
            _, arr = data[variant]
            n_tail = max(arr.shape[1] // 10, 1)
            per_seed = arr[:, -n_tail:].mean(axis=1)
            if variant not in results:
                results[variant] = ([], [])
            results[variant][0].append(per_seed.mean())
            results[variant][1].append(per_seed.std())

    if not hidden_vals or not results:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in ["vanilla", "modal_action"]:
        if variant not in results:
            continue
        style = VARIANT_STYLES.get(variant, {"color": "gray", "label": variant})
        means, stds = results[variant]
        ax.errorbar(hidden_vals, means, yerr=stds, marker="o", capsize=5,
                     color=style["color"], label=style["label"], linewidth=2, markersize=8)

    ax.set_xlabel("Hidden Dimension", fontsize=12)
    ax.set_ylabel("Final Tracking RMSE (rad)", fontsize=12)
    ax.set_title("Modal Action Advantage vs Network Capacity", fontsize=14)
    ax.set_xticks(hidden_vals)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "toy_capacity_sweep.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------

def print_summary(histories: dict) -> None:
    data = _extract_metric(histories, "mean_rmse")
    if not data:
        print("No data found.")
        return

    print("\n" + "=" * 70)
    print("  TOY DUAL-ARM VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  {'Variant':<22} {'Final RMSE':>12} {'Steps→{:.2f}'.format(RMSE_THRESHOLD):>14} {'Seeds':>6}")
    print("-" * 70)

    for variant in ["vanilla", "coupling_features", "modal_action"]:
        if variant not in data:
            continue
        steps, arr = data[variant]
        n_tail = max(arr.shape[1] // 10, 1)
        final = arr[:, -n_tail:].mean(axis=1)

        steps_to = []
        for seed_arr in arr:
            below = np.where(seed_arr < RMSE_THRESHOLD)[0]
            steps_to.append(steps[below[0]] if len(below) > 0 else float("nan"))
        steps_to = np.array(steps_to)

        style = VARIANT_STYLES.get(variant, {"label": variant})
        print(f"  {style['label']:<22} {final.mean():.4f}±{final.std():.4f}"
              f"  {np.nanmean(steps_to):>10.0f}±{np.nanstd(steps_to):.0f}"
              f"  {arr.shape[0]:>5}")

    # Speedup
    if "vanilla" in data and "modal_action" in data:
        steps_v, arr_v = data["vanilla"]
        steps_m, arr_m = data["modal_action"]

        def mean_threshold_steps(steps, arr):
            s = []
            for sa in arr:
                below = np.where(sa < RMSE_THRESHOLD)[0]
                s.append(steps[below[0]] if len(below) > 0 else steps[-1])
            return np.mean(s)

        sv = mean_threshold_steps(steps_v, arr_v)
        sm = mean_threshold_steps(steps_m, arr_m)
        print("-" * 70)
        print(f"  Modal speedup: {sv / max(sm, 1):.2f}x")
    print("=" * 70)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main(save_dir: str = "results/toy", fig_dir: str = FIG_DIR) -> None:
    histories = load_histories(save_dir)
    if histories:
        print_summary(histories)
        plot_learning_curves(histories, fig_dir)
        plot_sample_efficiency(histories, fig_dir)
        plot_final_comparison(histories, fig_dir)
    else:
        print(f"No base results in {save_dir}/ (checking sweep dirs...)")

    # Sweep plots look for subdirectories; run even without base results
    plot_mass_sweep(save_dir, fig_dir)
    plot_capacity_sweep(save_dir, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="results/toy")
    parser.add_argument("--fig-dir", type=str, default=FIG_DIR)
    args = parser.parse_args()
    main(save_dir=args.save_dir, fig_dir=args.fig_dir)
