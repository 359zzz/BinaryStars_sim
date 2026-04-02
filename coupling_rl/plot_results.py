"""Plot coupling RL results (Scheme beta).

Key figures:
1. Learning curves: reward vs steps for 5 variants
2. Final performance comparison
3. Coupling correlation: J_ij and C_ij vs action MI

Usage:
    python -m coupling_rl.plot_results --save_dir results/beta
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
    "vanilla":        {"color": "C0", "label": "Vanilla PPO"},
    "geometric":      {"color": "C1", "label": "Geometric PPO"},
    "coupling":       {"color": "C2", "label": "Coupling PPO (|J|)"},
    "quantum_c":      {"color": "C3", "label": "Quantum PPO (C_ij)"},
    "quantum_decomp": {"color": "C4", "label": "Quantum Decomp PPO"},
}


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_histories(save_dir: str) -> dict[str, list[list[dict]]]:
    """Load all training histories grouped by variant."""
    results = {}
    for d in sorted(Path(save_dir).iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        variant = name.rsplit("_seed", 1)[0]
        hist_path = d / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            results.setdefault(variant, []).append(hist)
    return results


def plot_learning_curves(
    histories: dict[str, list[list[dict]]],
    fig_dir: str = "figures",
) -> None:
    """Plot reward learning curves with mean +/- std across seeds."""
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for variant, runs in histories.items():
        style = VARIANT_STYLES.get(variant, {"color": "C5", "label": variant})

        all_rewards = []
        all_success = []
        for run in runs:
            all_rewards.append([e["mean_reward"] for e in run])
            all_success.append([e["success_rate"] for e in run])

        min_len = min(len(r) for r in all_rewards)
        rewards_arr = np.array([r[:min_len] for r in all_rewards])
        success_arr = np.array([s[:min_len] for s in all_success])
        steps = np.array([e["step"] for e in runs[0][:min_len]])

        window = 10
        for ax, data, title, ylabel in [
            (axes[0], rewards_arr, "Episode Reward", "Mean Reward"),
            (axes[1], success_arr, "Success Rate", "Success Rate"),
        ]:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            mean_s = smooth(mean, window)
            std_s = smooth(std, window)
            steps_s = steps[:len(mean_s)]

            ax.plot(steps_s, mean_s, color=style["color"], label=style["label"])
            ax.fill_between(
                steps_s, mean_s - std_s, mean_s + std_s,
                alpha=0.15, color=style["color"],
            )
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle("Coupling-Aware RL: 5 PPO Variants", fontsize=14)
    plt.tight_layout()

    path = os.path.join(fig_dir, "beta_learning_curves.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_final_comparison(
    histories: dict[str, list[list[dict]]],
    fig_dir: str = "figures",
) -> None:
    """Bar chart of final performance."""
    os.makedirs(fig_dir, exist_ok=True)

    variant_order = ["vanilla", "geometric", "coupling", "quantum_c", "quantum_decomp"]
    variants = []
    final_rewards = []
    final_success = []

    for variant in variant_order:
        if variant not in histories:
            continue
        runs = histories[variant]
        rewards = []
        success = []
        for run in runs:
            last = run[-20:] if len(run) >= 20 else run
            rewards.append(np.mean([e["mean_reward"] for e in last]))
            success.append(np.mean([e["success_rate"] for e in last]))

        style = VARIANT_STYLES.get(variant, {"label": variant})
        variants.append(style["label"])
        final_rewards.append((np.mean(rewards), np.std(rewards)))
        final_success.append((np.mean(success), np.std(success)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(variants))
    colors = [VARIANT_STYLES.get(v, {"color": "C5"})["color"] for v in variant_order if v in histories]

    for ax, data, title in [
        (axes[0], final_rewards, "Final Reward"),
        (axes[1], final_success, "Final Success Rate"),
    ]:
        means = [d[0] for d in data]
        stds = [d[1] for d in data]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(fig_dir, "beta_final_comparison.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_coupling_correlation(
    policy_path: str,
    fig_dir: str = "figures",
    n_episodes: int = 50,
    variant: str = "vanilla",
) -> None:
    """Correlation analysis: J_ij and C_ij vs MI(a_i, a_j).

    Compares classical (|J|) and quantum (C_ij) as predictors of learned
    action coordination.
    """
    os.makedirs(fig_dir, exist_ok=True)

    import torch
    from coupling_rl.networks import VanillaPolicy, CouplingAwarePolicy
    from envs.openarm_reach import OpenArmReachEnv
    from physics.openarm_params import (
        compute_openarm_coupling,
        compute_openarm_mass_matrix,
    )
    from quantum_prior.entanglement_graph import (
        compute_entanglement_graph,
    )

    # Load policy — match architecture used during training
    from coupling_rl.networks import QuantumDecomposedPolicy
    if variant in ("coupling", "quantum_c"):
        policy = CouplingAwarePolicy(obs_dim=41)
    elif variant == "quantum_decomp":
        policy = QuantumDecomposedPolicy(obs_dim=41)
    elif variant == "geometric":
        from coupling_rl.networks import GeometricPolicy
        policy = GeometricPolicy(obs_dim=20)
    else:
        policy = VanillaPolicy(obs_dim=20)
    policy.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
    policy.eval()

    env = OpenArmReachEnv()
    all_j = []
    all_c = []
    all_actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(200):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.get_action(obs_t)
            action_np = action[0].numpy()
            q = obs[:7]

            J = compute_openarm_coupling(q)
            M = compute_openarm_mass_matrix(q)
            C = compute_entanglement_graph(M)

            all_j.append(J)
            all_c.append(C)
            all_actions.append(action_np)

            obs, _, term, trunc, _ = env.step(action_np * 50.0)
            if term or trunc:
                break

    env.close()

    actions = np.array(all_actions)
    j_matrices = np.array(all_j)
    c_matrices = np.array(all_c)

    n_joints = 7
    j_means = []
    c_means = []
    action_corrs = []

    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            j_mean = np.mean(np.abs(j_matrices[:, i, j]))
            c_mean = np.mean(c_matrices[:, i, j])
            corr = abs(np.corrcoef(actions[:, i], actions[:, j])[0, 1])
            j_means.append(j_mean)
            c_means.append(c_mean)
            action_corrs.append(corr)

    j_means = np.array(j_means)
    c_means = np.array(c_means)
    action_corrs = np.array(action_corrs)
    r_j = np.corrcoef(j_means, action_corrs)[0, 1]
    r_c = np.corrcoef(c_means, action_corrs)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, x_data, x_label, r_val, title in [
        (axes[0], j_means, "|J_ij| (classical)", r_j, "Classical Coupling"),
        (axes[1], c_means, "C_ij (quantum)", r_c, "Quantum Entanglement"),
    ]:
        ax.scatter(x_data, action_corrs, alpha=0.6, s=40)
        ax.set_xlabel(x_label)
        ax.set_ylabel("|corr(a_i, a_j)|")
        ax.set_title(f"{title} vs Action Coordination (r={r_val:.3f})")
        ax.grid(True, alpha=0.3)

        idx = 0
        for i in range(n_joints):
            for j in range(i + 1, n_joints):
                ax.annotate(f"({i},{j})", (x_data[idx], action_corrs[idx]),
                            fontsize=6, alpha=0.5)
                idx += 1

    plt.tight_layout()
    path = os.path.join(fig_dir, "beta_coupling_correlation.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    print(f"Classical r(|J|, MI) = {r_j:.3f}")
    print(f"Quantum   r(C_ij, MI) = {r_c:.3f}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results/beta")
    parser.add_argument("--fig_dir", type=str, default="figures")
    parser.add_argument("--fallback_policy", type=str, default=None,
                        help="Path to trained policy for correlation analysis")
    parser.add_argument("--variant", type=str, default="vanilla")
    args = parser.parse_args()

    histories = load_histories(args.save_dir)
    if histories:
        plot_learning_curves(histories, args.fig_dir)
        plot_final_comparison(histories, args.fig_dir)
    else:
        print("No training histories found.")

    if args.fallback_policy:
        plot_coupling_correlation(
            args.fallback_policy, args.fig_dir,
            variant=args.variant,
        )


if __name__ == "__main__":
    main()
