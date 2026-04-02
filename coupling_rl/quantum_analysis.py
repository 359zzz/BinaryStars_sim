"""Scheme delta: Posterior analysis of quantum structure emergence.

After training, analyze whether learned policies exhibit quantum (vs classical)
coordination structure. Key metric: r(MI(a_i, a_j), C_ij) vs r(MI(a_i, a_j), |J_ij|).

If r_quantum > r_classical, the quantum structure prior better predicts learned behavior.
Even vanilla PPO (no prior) may show emergent quantum structure.

Usage:
    python -m coupling_rl.quantum_analysis --policy results/beta/vanilla_seed0/policy.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from coupling_rl.networks import (
    CouplingAwarePolicy,
    VanillaPolicy,
    GeometricPolicy,
    QuantumDecomposedPolicy,
)
from envs.openarm_reach import OpenArmReachEnv
from physics.openarm_params import (
    N_JOINTS,
    compute_openarm_coupling,
    compute_openarm_mass_matrix,
)
from quantum_prior.entanglement_graph import (
    compute_entanglement_graph,
)


def compute_mutual_information_proxy(
    actions: np.ndarray,
) -> np.ndarray:
    """Compute pairwise action correlation matrix as MI proxy.

    Returns (n, n) matrix where entry (i,j) = |corr(a_i, a_j)|.
    """
    n = actions.shape[1]
    corr = np.abs(np.corrcoef(actions.T))
    np.fill_diagonal(corr, 0.0)
    return corr


def quantum_vs_classical_correlation(
    policy_path: str,
    variant: str = "vanilla",
    n_episodes: int = 100,
    device: str = "cpu",
) -> dict:
    """Post-training analysis: which structure predicts action coordination?

    For each joint pair (i,j):
    - Compute mean |J_ij| (classical) and mean C_ij (quantum) across configs
    - Compute MI(a_i, a_j) proxy from action correlations
    - Compare r(MI, J) vs r(MI, C)

    Returns dict with correlations, per-pair data, and significance.
    """
    # Load policy — match architecture used during training
    if variant in ("coupling", "quantum_c"):
        policy = CouplingAwarePolicy(obs_dim=41)
    elif variant == "quantum_decomp":
        policy = QuantumDecomposedPolicy(obs_dim=41)
    elif variant == "geometric":
        policy = GeometricPolicy(obs_dim=20)
    else:
        policy = VanillaPolicy(obs_dim=20)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    env = OpenArmReachEnv()

    # Collect trajectories
    all_j_upper = []  # per-step upper-triangle |J_ij|
    all_c_upper = []  # per-step upper-triangle C_ij
    all_actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 17)
        for _ in range(200):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.get_action(obs_t)
            action_np = action[0].numpy()
            q = obs[:7]

            J = compute_openarm_coupling(q)
            M = compute_openarm_mass_matrix(q)
            C = compute_entanglement_graph(M)

            # Extract upper triangle
            j_upper = []
            c_upper = []
            for i in range(N_JOINTS):
                for j in range(i + 1, N_JOINTS):
                    j_upper.append(abs(J[i, j]))
                    c_upper.append(C[i, j])

            all_j_upper.append(j_upper)
            all_c_upper.append(c_upper)
            all_actions.append(action_np)

            obs, _, term, trunc, _ = env.step(action_np * 50.0)
            if term or trunc:
                break

    env.close()

    actions = np.array(all_actions)
    j_data = np.array(all_j_upper)
    c_data = np.array(all_c_upper)

    # Mean coupling/entanglement across trajectory
    j_means = j_data.mean(axis=0)  # (21,)
    c_means = c_data.mean(axis=0)  # (21,)

    # Pairwise action MI proxy
    mi_matrix = compute_mutual_information_proxy(actions)
    mi_upper = []
    pair_labels = []
    for i in range(N_JOINTS):
        for j in range(i + 1, N_JOINTS):
            mi_upper.append(mi_matrix[i, j])
            pair_labels.append(f"({i},{j})")
    mi_upper = np.array(mi_upper)

    # Correlations
    r_classical = float(np.corrcoef(j_means, mi_upper)[0, 1])
    r_quantum = float(np.corrcoef(c_means, mi_upper)[0, 1])

    # Identify indirect coupling pairs (smoking gun)
    indirect_pairs = []
    for idx, (jv, cv) in enumerate(zip(j_means, c_means)):
        if jv < 0.01 and cv > 0.05:
            indirect_pairs.append({
                "pair": pair_labels[idx],
                "J": float(jv),
                "C": float(cv),
                "MI": float(mi_upper[idx]),
            })

    result = {
        "variant": variant,
        "n_episodes": n_episodes,
        "r_classical": r_classical,
        "r_quantum": r_quantum,
        "quantum_advantage": r_quantum > r_classical,
        "indirect_coupling_pairs": indirect_pairs,
        "per_pair": {
            pair_labels[k]: {
                "J_mean": float(j_means[k]),
                "C_mean": float(c_means[k]),
                "MI": float(mi_upper[k]),
            }
            for k in range(len(pair_labels))
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--variant", type=str, default="vanilla")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Analyzing {args.variant} policy: {args.policy}")
    result = quantum_vs_classical_correlation(
        args.policy, args.variant, args.n_episodes,
    )

    print(f"\n=== Scheme delta Results ===")
    print(f"  r(MI, |J|) = {result['r_classical']:.3f}  (classical)")
    print(f"  r(MI, C)   = {result['r_quantum']:.3f}  (quantum)")
    print(f"  Quantum advantage: {result['quantum_advantage']}")

    if result["indirect_coupling_pairs"]:
        print(f"\n  Indirect coupling pairs ({len(result['indirect_coupling_pairs'])}):")
        for p in result["indirect_coupling_pairs"]:
            print(f"    {p['pair']}: J={p['J']:.4f}, C={p['C']:.4f}, MI={p['MI']:.4f}")

    out_path = args.output or f"results/delta_{args.variant}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
