#!/usr/bin/env python3
"""Analyze Scheme beta + delta results → paper Table + figures.

Usage:
    python scripts/analyze_beta.py --export-dir beta_export/beta_export
    python scripts/analyze_beta.py --export-dir beta_export/beta_export --latex
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_ind, mannwhitneyu

rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

VARIANT_ORDER = ["vanilla", "geometric", "coupling", "quantum_c", "quantum_decomp"]
VARIANT_LABELS = {
    "vanilla": "Vanilla PPO",
    "geometric": "Geometric PPO",
    "coupling": "Coupling PPO ($|J|$)",
    "quantum_c": "Quantum PPO ($C_{ij}$)",
    "quantum_decomp": "Quantum Decomp PPO",
}
VARIANT_COLORS = {
    "vanilla": "#377eb8",
    "geometric": "#ff7f00",
    "coupling": "#4daf4a",
    "quantum_c": "#984ea3",
    "quantum_decomp": "#e41a1c",
}


# ── Load data ────────────────────────────────────────────────────────────────

def load_delta_results(delta_dir: Path) -> dict[str, list[dict]]:
    """Load delta JSON files grouped by variant."""
    results = defaultdict(list)
    for f in sorted(delta_dir.glob("delta_*_seed*.json")):
        with open(f) as fh:
            data = json.load(fh)
        variant = data["variant"]
        results[variant].append(data)
    return dict(results)


def load_learning_curves(raw_dir: Path) -> dict[str, list[list[dict]]]:
    """Load training history grouped by variant."""
    results = defaultdict(list)
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        variant = d.name.rsplit("_seed", 1)[0]
        hist_path = d / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                results[variant].append(json.load(f))
    return dict(results)


# ── Delta analysis ───────────────────────────────────────────────────────────

def compute_delta_table(delta: dict[str, list[dict]]) -> list[dict]:
    """Compute mean±std for r_classical and r_quantum per variant."""
    rows = []
    for v in VARIANT_ORDER:
        if v not in delta:
            continue
        runs = delta[v]
        r_cl = [r["r_classical"] for r in runs]
        r_qu = [r["r_quantum"] for r in runs]
        qa = sum(1 for r in runs if r["r_quantum"] > r["r_classical"])

        rows.append({
            "variant": v,
            "label": VARIANT_LABELS[v],
            "r_classical_mean": np.mean(r_cl),
            "r_classical_std": np.std(r_cl),
            "r_quantum_mean": np.mean(r_qu),
            "r_quantum_std": np.std(r_qu),
            "qa_wins": qa,
            "n_seeds": len(runs),
            "r_classical_all": r_cl,
            "r_quantum_all": r_qu,
        })
    return rows


def compute_significance(delta: dict[str, list[dict]]) -> list[dict]:
    """Statistical tests: quantum_decomp vs each other variant."""
    if "quantum_decomp" not in delta:
        return []
    qd_rq = [r["r_quantum"] for r in delta["quantum_decomp"]]
    tests = []
    for v in VARIANT_ORDER:
        if v == "quantum_decomp" or v not in delta:
            continue
        other_rq = [r["r_quantum"] for r in delta[v]]
        t_stat, p_val = ttest_ind(qd_rq, other_rq, equal_var=False)
        u_stat, u_p = mannwhitneyu(qd_rq, other_rq, alternative="greater")
        tests.append({
            "comparison": f"quantum_decomp vs {v}",
            "t_stat": float(t_stat),
            "t_p": float(p_val),
            "u_stat": float(u_stat),
            "u_p": float(u_p),
        })
    return tests


def compute_final_performance(histories: dict[str, list[list[dict]]]) -> list[dict]:
    """Final reward and success rate per variant."""
    rows = []
    for v in VARIANT_ORDER:
        if v not in histories:
            continue
        final_rewards = []
        final_success = []
        for hist in histories[v]:
            last = hist[-20:] if len(hist) >= 20 else hist
            final_rewards.append(np.mean([e["mean_reward"] for e in last]))
            final_success.append(np.mean([e["success_rate"] for e in last]))
        rows.append({
            "variant": v,
            "label": VARIANT_LABELS[v],
            "reward_mean": np.mean(final_rewards),
            "reward_std": np.std(final_rewards),
            "success_mean": np.mean(final_success),
            "success_std": np.std(final_success),
            "n_seeds": len(final_rewards),
        })
    return rows


# ── LaTeX output ─────────────────────────────────────────────────────────────

def format_latex_delta_table(rows: list[dict], tests: list[dict]) -> str:
    """Paper Table: Scheme delta — quantum vs classical structure alignment."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Post-training structure analysis (Scheme~$\delta$). "
        r"$r(\mathrm{MI}, |J|)$: correlation between action mutual information "
        r"and classical coupling. $r(\mathrm{MI}, C)$: correlation with quantum "
        r"entanglement. QA: quantum advantage (seeds with $r_C > r_J$).}",
        r"\label{tab:delta}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Variant & $r(\mathrm{MI}, |J|)$ & $r(\mathrm{MI}, C)$ & QA & Seeds \\",
        r"\hline",
    ]
    for row in rows:
        v = row["label"]
        rcl = f"{row['r_classical_mean']:+.2f} $\\pm$ {row['r_classical_std']:.2f}"
        rqu = f"{row['r_quantum_mean']:+.2f} $\\pm$ {row['r_quantum_std']:.2f}"
        qa = f"{row['qa_wins']}/{row['n_seeds']}"
        # Bold the best r_quantum
        if row["variant"] == "quantum_decomp":
            rqu = r"\textbf{" + rqu + "}"
        lines.append(f"  {v} & {rcl} & {rqu} & {qa} & {row['n_seeds']} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    # Add significance footnote
    if tests:
        qd_vs_all = [t for t in tests]
        p_vals = [f"{t['comparison'].split('vs ')[1]}: $p={t['t_p']:.3f}$" for t in qd_vs_all]
        lines.append(r"\vspace{2pt}")
        lines.append(r"\footnotesize{Welch's $t$-test (quantum\_decomp vs others): " +
                      ", ".join(p_vals) + "}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def format_latex_performance_table(perf: list[dict]) -> str:
    """Paper Table: final training performance."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Final training performance (Scheme~$\beta$, last 20 evaluations).}",
        r"\label{tab:beta-performance}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Variant & Reward & Success Rate & Seeds \\",
        r"\hline",
    ]
    for row in perf:
        rew = f"{row['reward_mean']:.1f} $\\pm$ {row['reward_std']:.1f}"
        suc = f"{row['success_mean']:.2f} $\\pm$ {row['success_std']:.2f}"
        lines.append(f"  {row['label']} & {rew} & {suc} & {row['n_seeds']} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_delta_comparison(rows: list[dict], fig_dir: Path):
    """Fig: r(MI, |J|) vs r(MI, C) grouped bar chart per variant."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    variants = [r["variant"] for r in rows]
    x = np.arange(len(variants))
    w = 0.35

    r_cl_means = [r["r_classical_mean"] for r in rows]
    r_cl_stds = [r["r_classical_std"] for r in rows]
    r_qu_means = [r["r_quantum_mean"] for r in rows]
    r_qu_stds = [r["r_quantum_std"] for r in rows]

    ax.bar(x - w/2, r_cl_means, w, yerr=r_cl_stds, capsize=3,
           label=r"$r(\mathrm{MI}, |J|)$ (classical)", color="#377eb8",
           edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.bar(x + w/2, r_qu_means, w, yerr=r_qu_stds, capsize=3,
           label=r"$r(\mathrm{MI}, C)$ (quantum)", color="#e41a1c",
           edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v].replace(" PPO", "").replace("$", "")
                         for v in variants], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Correlation with action MI")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.3, 0.8)

    fig.tight_layout()
    path = fig_dir / "delta_comparison.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_delta_scatter(rows: list[dict], fig_dir: Path):
    """Fig: scatter all seeds, r_classical vs r_quantum, colored by variant."""
    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    for row in rows:
        v = row["variant"]
        r_cl = row["r_classical_all"]
        r_qu = row["r_quantum_all"]
        ax.scatter(r_cl, r_qu, s=40, c=VARIANT_COLORS[v],
                   edgecolors="black", linewidths=0.5,
                   label=VARIANT_LABELS[v], zorder=3, alpha=0.8)

    # Diagonal: QA boundary
    lim = [-0.6, 0.8]
    ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
    ax.fill_between(lim, lim, [lim[1], lim[1]], alpha=0.05, color="red")
    ax.text(0.05, 0.95, "Quantum\nadvantage", transform=ax.transAxes,
            va="top", fontsize=7, color="red", alpha=0.6)

    ax.set_xlabel(r"$r(\mathrm{MI}, |J|)$ (classical)")
    ax.set_ylabel(r"$r(\mathrm{MI}, C)$ (quantum)")
    ax.set_xlim(-0.6, 0.7)
    ax.set_ylim(-0.4, 0.8)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = fig_dir / "delta_scatter.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_learning_curves(histories: dict[str, list[list[dict]]], fig_dir: Path):
    """Fig: learning curves with mean±std shading."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for v in VARIANT_ORDER:
        if v not in histories:
            continue
        runs = histories[v]
        all_rewards = [[e["mean_reward"] for e in run] for run in runs]
        all_success = [[e["success_rate"] for e in run] for run in runs]

        min_len = min(len(r) for r in all_rewards)
        rewards = np.array([r[:min_len] for r in all_rewards])
        success = np.array([s[:min_len] for s in all_success])
        steps = np.array([e["step"] for e in runs[0][:min_len]])

        color = VARIANT_COLORS[v]
        label = VARIANT_LABELS[v]

        for ax, data in [(axes[0], rewards), (axes[1], success)]:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # Smooth
            w = min(10, len(mean) // 5)
            if w > 1:
                kernel = np.ones(w) / w
                mean_s = np.convolve(mean, kernel, mode="valid")
                std_s = np.convolve(std, kernel, mode="valid")
                steps_s = steps[:len(mean_s)]
            else:
                mean_s, std_s, steps_s = mean, std, steps

            ax.plot(steps_s, mean_s, color=color, label=label, linewidth=1)
            ax.fill_between(steps_s, mean_s - std_s, mean_s + std_s,
                            alpha=0.12, color=color)

    axes[0].set_ylabel("Episode Reward")
    axes[1].set_ylabel("Success Rate")
    for ax in axes:
        ax.set_xlabel("Environment Steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    path = fig_dir / "beta_learning_curves.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze beta + delta results")
    parser.add_argument("--export-dir", default="beta_export/beta_export")
    parser.add_argument("--raw-dir", default=None,
                        help="Direct path to training results (e.g. results/beta)")
    parser.add_argument("--output", default="results/beta_analysis.json")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    export = Path(args.export_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data — raw-dir takes precedence over export-dir/raw
    raw_dir = Path(args.raw_dir) if args.raw_dir else export / "raw"
    delta_dir = export / "delta"

    delta = load_delta_results(delta_dir) if delta_dir.exists() else {}
    if raw_dir.exists():
        histories = load_learning_curves(raw_dir)
    else:
        histories = {}

    print(f"Loaded: {sum(len(v) for v in delta.values())} delta results, "
          f"{sum(len(v) for v in histories.values())} training histories")

    # Compute tables
    delta_table = compute_delta_table(delta)
    significance = compute_significance(delta)
    performance = compute_final_performance(histories)

    # Print summary
    print("\n=== Scheme delta: Structure Alignment ===")
    print(f"{'Variant':<22s} {'r(MI,|J|)':>12s} {'r(MI,C)':>12s} {'QA':>5s}")
    print("-" * 55)
    for row in delta_table:
        rcl = f"{row['r_classical_mean']:+.2f}±{row['r_classical_std']:.2f}"
        rqu = f"{row['r_quantum_mean']:+.2f}±{row['r_quantum_std']:.2f}"
        qa = f"{row['qa_wins']}/{row['n_seeds']}"
        print(f"  {row['label']:<20s} {rcl:>12s} {rqu:>12s} {qa:>5s}")

    print("\n=== Significance (quantum_decomp vs others) ===")
    for t in significance:
        sig = "**" if t["t_p"] < 0.01 else "*" if t["t_p"] < 0.05 else ""
        print(f"  {t['comparison']:<35s} t={t['t_stat']:+.2f}  p={t['t_p']:.4f} {sig}")

    print("\n=== Final Training Performance ===")
    print(f"{'Variant':<22s} {'Reward':>14s} {'Success':>14s}")
    print("-" * 55)
    for row in performance:
        rew = f"{row['reward_mean']:.1f}±{row['reward_std']:.1f}"
        suc = f"{row['success_mean']:.2f}±{row['success_std']:.2f}"
        print(f"  {row['label']:<20s} {rew:>14s} {suc:>14s}")

    # LaTeX
    if args.latex:
        print("\n" + "=" * 60)
        print(format_latex_delta_table(delta_table, significance))
        print()
        print(format_latex_performance_table(performance))

    # Figures
    plot_delta_comparison(delta_table, fig_dir)
    plot_delta_scatter(delta_table, fig_dir)
    if histories:
        plot_learning_curves(histories, fig_dir)

    # Save JSON
    analysis = {
        "delta_table": [{k: v for k, v in row.items() if k != "r_classical_all" and k != "r_quantum_all"}
                        for row in delta_table],
        "delta_per_seed": {row["variant"]: {
            "r_classical": row["r_classical_all"],
            "r_quantum": row["r_quantum_all"],
        } for row in delta_table},
        "significance": significance,
        "performance": performance,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis: {args.output}")


if __name__ == "__main__":
    main()
