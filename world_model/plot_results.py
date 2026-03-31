"""Plot world model transfer results (Scheme gamma).

Key figures:
1. Transfer curve: x = fine-tune data, y = prediction error
2. Indirect coupling comparison: C-MLP vs J-MLP vs MLP on smoking gun pairs
3. Training summary

Usage:
    python -m world_model.plot_results --results results/gamma/transfer_results.json
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_STYLES = {
    "mlp":   {"color": "C0", "marker": "o", "label": "MLP"},
    "j_mlp": {"color": "C1", "marker": "s", "label": "J-MLP (classical)"},
    "c_mlp": {"color": "C2", "marker": "D", "label": "C-MLP (quantum)"},
    "delan":  {"color": "C3", "marker": "^", "label": "DeLaN"},
}


def plot_transfer(results: dict, save_dir: str = "figures") -> None:
    """Generate the key transfer figure (5 models)."""
    os.makedirs(save_dir, exist_ok=True)

    finetune_sizes = [0, 10, 50, 100, 500, 1000, 5000]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric_key, title in [
        (axes[0], "total_rmse", "1-Step RMSE"),
        (axes[1], "multi_step_rmse", "50-Step Rollout RMSE"),
        (axes[2], "mean_indirect_rmse", "Indirect Coupling RMSE"),
    ]:
        for payload_key in sorted(results.keys()):
            pr = results[payload_key]
            payload = pr["payload_kg"]
            if payload == 0.0:
                continue  # Skip in-distribution

            # CRBA horizontal line
            if metric_key in pr.get("crba", {}):
                crba_val = pr["crba"][metric_key]
                ax.axhline(crba_val, linestyle="--", alpha=0.5, color="gray",
                           label=f"CRBA {payload}kg" if ax == axes[0] else "")

            # Neural model curves
            for model_name, style in MODEL_STYLES.items():
                vals = []
                for n_ft in finetune_sizes:
                    key = f"{model_name}_ft_{n_ft}"
                    if key in pr and metric_key in pr[key]:
                        vals.append(pr[key][metric_key])
                    else:
                        vals.append(np.nan)

                if all(np.isnan(v) for v in vals):
                    continue

                label = (
                    f"{style['label']} {payload}kg"
                    if ax == axes[0] else ""
                )
                ax.plot(
                    finetune_sizes, vals,
                    f"{style['marker']}-", color=style["color"],
                    label=label, alpha=0.8,
                )

        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Fine-tune samples")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("World Model Transfer: Quantum vs Classical Prior", fontsize=14)
    plt.tight_layout()

    path = os.path.join(save_dir, "gamma_transfer.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_indirect_comparison(results: dict, save_dir: str = "figures") -> None:
    """Bar chart: indirect coupling RMSE per model at small data regime."""
    os.makedirs(save_dir, exist_ok=True)

    # Use N=100 fine-tune as "small data" representative
    n_ft = 100
    model_names = ["mlp", "j_mlp", "c_mlp", "delan"]
    labels = ["MLP", "J-MLP", "C-MLP", "DeLaN"]
    colors = ["C0", "C1", "C2", "C3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(model_names))
    width = 0.35

    for pidx, payload_key in enumerate(sorted(results.keys())):
        pr = results[payload_key]
        if pr["payload_kg"] == 0.0:
            continue

        vals = []
        for mn in model_names:
            key = f"{mn}_ft_{n_ft}"
            if key in pr and "mean_indirect_rmse" in pr[key]:
                vals.append(pr[key]["mean_indirect_rmse"])
            else:
                vals.append(0)

        offset = (pidx - 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=f"{pr['payload_kg']}kg",
               color=colors, alpha=0.7 + 0.15 * pidx)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Indirect Coupling RMSE")
    ax.set_title(f"Smoking Gun: Indirect Coupling Pairs (N={n_ft} fine-tune)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "gamma_indirect_coupling.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_training_curves(save_dir: str = "results/gamma", fig_dir: str = "figures") -> None:
    """Plot training loss summary."""
    os.makedirs(fig_dir, exist_ok=True)

    meta_path = os.path.join(save_dir, "train_meta.json")
    if not os.path.exists(meta_path):
        print("No training metadata found.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    lines = []
    for key in ["mlp", "j_mlp", "c_mlp", "delan"]:
        loss_key = f"{key}_final_loss"
        time_key = f"{key}_time_s"
        if loss_key in meta:
            lines.append(f"{key.upper()}: loss={meta[loss_key]:.6f} ({meta[time_key]:.0f}s)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "\n".join(lines),
            ha="center", va="center", fontsize=12, transform=ax.transAxes,
            family="monospace")
    ax.set_title("Training Summary")
    ax.set_axis_off()

    path = os.path.join(fig_dir, "gamma_training.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/gamma/transfer_results.json")
    parser.add_argument("--save_dir", type=str, default="figures")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    plot_transfer(results, args.save_dir)
    plot_indirect_comparison(results, args.save_dir)
    plot_training_curves(
        save_dir=os.path.dirname(args.results),
        fig_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
