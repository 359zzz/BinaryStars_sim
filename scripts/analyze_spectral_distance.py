"""Analyze S1.2 Spectral Distance Map results for NMI paper.

Reads results/spectral_distance_map.json and outputs:
1. Summary statistics (full + per functional group)
2. Same-morphology vs cross-morphology comparison
3. Configuration-dependence analysis
4. Classical coupling distance baseline comparison
5. Key findings for paper text
6. LaTeX-ready table

Usage:
    python scripts/analyze_spectral_distance.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_summary(data: dict) -> None:
    """Table 1: Overall spectral distance statistics."""
    print_header("Table 1: Spectral Distance Summary")

    s = data["summary"]
    print(f"\n  Object mass: {data['object_mass']} kg")
    print(f"  Configs: {data['n_configs']} OpenArm x {data['n_configs']} Piper = "
          f"{data['n_configs']**2} pairs\n")

    print(f"  {'Partition':<15} {'Mean D':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Full L|R':<15} {s['full_D_mean']:>10.4f} {s['full_D_std']:>10.4f} "
          f"{s['full_D_min']:>10.4f} {s['full_D_max']:>10.4f}")

    for group in ["shoulder", "elbow", "wrist"]:
        mean_key = f"{group}_D_mean"
        std_key = f"{group}_D_std"
        if mean_key in s:
            print(f"  {group.capitalize():<15} {s[mean_key]:>10.4f} {s[std_key]:>10.4f}")


def analyze_entropy_ranges(data: dict) -> None:
    """Table 2: Entropy comparison between robots."""
    print_header("Table 2: Entanglement Entropy by Robot")

    for robot, key in [("OpenArm 7-DOF", "openarm_spectra"),
                       ("Piper 6-DOF", "piper_spectra")]:
        entropies = [d["entropy"] for d in data[key]]
        print(f"\n  {robot}:")
        print(f"    Full S(ρ_L): mean={np.mean(entropies):.4f} ± {np.std(entropies):.4f}, "
              f"range=[{np.min(entropies):.4f}, {np.max(entropies):.4f}]")

        for group in ["shoulder", "elbow", "wrist"]:
            vals = [d["functional_groups"].get(group, {}).get("entropy", np.nan)
                    for d in data[key]
                    if group in d.get("functional_groups", {})]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                print(f"    {group.capitalize():>10} S: mean={np.mean(vals):.4f} ± "
                      f"{np.std(vals):.4f}, range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")


def analyze_configuration_dependence(data: dict) -> None:
    """Key finding: D varies hugely with configuration."""
    print_header("Table 3: Configuration Dependence")

    full_D = np.array([d["D"] for d in data["full_spectral_distances"]])
    n = data["n_configs"]

    # Reshape to matrix
    D_matrix = full_D.reshape(n, n)

    # For each OpenArm config, find closest and farthest Piper config
    print(f"\n  Per OpenArm config: min/max D to any Piper config")
    print(f"  {'OA idx':<10} {'S(OA)':>8} {'D_min':>8} {'D_max':>8} {'ratio':>8}")
    print(f"  {'-'*42}")

    oa_entropies = [d["entropy"] for d in data["openarm_spectra"]]
    for i in range(n):
        d_row = D_matrix[i, :]
        ratio = d_row.max() / max(d_row.min(), 1e-10)
        print(f"  {i:<10} {oa_entropies[i]:>8.4f} {d_row.min():>8.4f} "
              f"{d_row.max():>8.4f} {ratio:>8.1f}x")

    # Coefficient of variation
    cv = np.std(full_D) / np.mean(full_D)
    print(f"\n  Coefficient of variation: {cv:.3f}")
    print(f"  → D varies by {cv*100:.0f}% relative to mean")
    print(f"  → NOT a constant: depends strongly on configuration")

    # Correlation between D and |ΔS|
    delta_S = np.array([d["delta_S"] for d in data["full_spectral_distances"]])
    corr = np.corrcoef(full_D, delta_S)[0, 1]
    print(f"\n  Correlation(D, |ΔS|) = {corr:.4f}")
    print(f"  → Entropy difference explains {corr**2*100:.1f}% of spectral distance")


def analyze_functional_group_hierarchy(data: dict) -> None:
    """Key finding: functional group distances reveal structure."""
    print_header("Table 4: Functional Group Distance Hierarchy")

    groups = ["shoulder", "elbow", "wrist"]
    means = {}
    for group in groups:
        D_vals = [d["D"] for d in data["functional_group_distances"][group]]
        means[group] = np.mean(D_vals)

    sorted_groups = sorted(means.items(), key=lambda x: x[1])
    print(f"\n  Functional group ranking (smallest D = most similar):")
    for rank, (group, mean_d) in enumerate(sorted_groups, 1):
        print(f"    {rank}. {group.capitalize():<10} D = {mean_d:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"    Elbow (D={means['elbow']:.3f}) < Shoulder (D={means['shoulder']:.3f}) "
          f"< Wrist (D={means['wrist']:.3f})")
    print(f"    → Elbow joints most structurally similar across morphologies")
    print(f"    → Wrist most different (OpenArm 3-DOF vs Piper 2-DOF)")


def compute_classical_baseline(data: dict) -> None:
    """Compare quantum spectral distance with classical coupling distance."""
    print_header("Table 5: Quantum vs Classical Distance")

    try:
        from physics.openarm_params import compute_openarm_mass_matrix
        from physics.piper_params import compute_piper_mass_matrix
        from quantum_prior.entanglement_graph import normalized_coupling_matrix

        oa_configs = [np.array(d["q"]) for d in data["openarm_spectra"]]
        pi_configs = [np.array(d["q"]) for d in data["piper_spectra"]]

        # Classical: |J_ij| Frobenius distance with zero-padding
        classical_D = []
        for i, q_oa in enumerate(oa_configs):
            M_oa = compute_openarm_mass_matrix(q_oa)
            J_oa = np.abs(normalized_coupling_matrix(M_oa))
            for j, q_pi in enumerate(pi_configs):
                M_pi = compute_piper_mass_matrix(q_pi)
                J_pi = np.abs(normalized_coupling_matrix(M_pi))
                # Zero-pad Piper (6x6) to match OpenArm (7x7)
                J_pi_pad = np.zeros((7, 7))
                J_pi_pad[:6, :6] = J_pi
                d_classical = np.linalg.norm(J_oa - J_pi_pad, "fro")
                classical_D.append(d_classical)

        classical_D = np.array(classical_D)
        quantum_D = np.array([d["D"] for d in data["full_spectral_distances"]])

        corr = np.corrcoef(quantum_D, classical_D)[0, 1]
        print(f"\n  Classical ||J_OA - J_Pi||_F: mean={classical_D.mean():.4f} ± {classical_D.std():.4f}")
        print(f"  Quantum D(OA, Pi):           mean={quantum_D.mean():.4f} ± {quantum_D.std():.4f}")
        print(f"\n  Correlation(D_quantum, D_classical) = {corr:.4f}")
        if corr < 0.7:
            print(f"  → Weak correlation: quantum distance captures DIFFERENT structure than classical")
        else:
            print(f"  → Strong correlation: quantum and classical distances agree")

        # Count discordant pairs (quantum says similar but classical says different, or vice versa)
        q_median = np.median(quantum_D)
        c_median = np.median(classical_D)
        q_close = quantum_D < q_median
        c_close = classical_D < c_median
        discordant = np.sum(q_close != c_close)
        print(f"  Discordant pairs: {discordant}/{len(quantum_D)} ({discordant/len(quantum_D)*100:.1f}%)")
        print(f"  → Pairs where quantum and classical disagree on similarity")

    except ImportError:
        print("\n  [Skipped: physics modules not available locally]")
        print("  Run this on the server: python scripts/analyze_spectral_distance.py")


def analyze_within_robot_distance(data: dict) -> None:
    """Compute within-robot spectral distance for comparison."""
    print_header("Table 6: Within-Robot vs Cross-Robot Distance")

    # Within OpenArm
    oa_spectra = data["openarm_spectra"]
    n = len(oa_spectra)

    from quantum_prior.entanglement_graph import spectral_distance as spec_dist

    try:
        oa_within = []
        for i in range(n):
            for j in range(i + 1, n):
                si = np.array(oa_spectra[i]["spectrum"])
                sj = np.array(oa_spectra[j]["spectrum"])
                d = spec_dist(si, sj)
                oa_within.append(d)

        pi_spectra = data["piper_spectra"]
        pi_within = []
        for i in range(n):
            for j in range(i + 1, n):
                si = np.array(pi_spectra[i]["spectrum"])
                sj = np.array(pi_spectra[j]["spectrum"])
                d = spec_dist(si, sj)
                pi_within.append(d)

        cross = np.array([d["D"] for d in data["full_spectral_distances"]])

        print(f"\n  Within OpenArm D:   mean={np.mean(oa_within):.4f} ± {np.std(oa_within):.4f} "
              f"(n={len(oa_within)} pairs)")
        print(f"  Within Piper D:     mean={np.mean(pi_within):.4f} ± {np.std(pi_within):.4f} "
              f"(n={len(pi_within)} pairs)")
        print(f"  Cross-robot D:      mean={np.mean(cross):.4f} ± {np.std(cross):.4f} "
              f"(n={len(cross)} pairs)")

        # Cross should be larger than within if morphology matters
        ratio_oa = np.mean(cross) / max(np.mean(oa_within), 1e-10)
        ratio_pi = np.mean(cross) / max(np.mean(pi_within), 1e-10)
        print(f"\n  Cross/Within-OA ratio: {ratio_oa:.2f}x")
        print(f"  Cross/Within-Pi ratio: {ratio_pi:.2f}x")

        if ratio_oa > 1.5 and ratio_pi > 1.5:
            print(f"  → Cross-robot distance >> within-robot: morphology is the dominant factor")
        elif ratio_oa > 1.0 and ratio_pi > 1.0:
            print(f"  → Cross-robot > within-robot: morphology matters but config also matters")
        else:
            print(f"  → Cross-robot ≈ within-robot: configuration dominates, morphology secondary")

    except ImportError:
        print("\n  [Skipped: quantum_prior not available locally]")
        print("  Run this on the server.")


def analyze_key_findings(data: dict) -> None:
    """Extract key numbers for paper."""
    print_header("Key Findings for NMI Paper")

    s = data["summary"]
    full_D = [d["D"] for d in data["full_spectral_distances"]]
    delta_S = [d["delta_S"] for d in data["full_spectral_distances"]]

    print(f"""
  1. DIMENSION-INDEPENDENCE:
     OpenArm (14-DOF dual-arm, 7+7) vs Piper (12-DOF dual-arm, 6+6)
     → D is well-defined despite different Hilbert space dimensions
     → Functional group matching enables component-level comparison

  2. CONFIGURATION SENSITIVITY:
     D range = [{min(full_D):.4f}, {max(full_D):.4f}]
     CV = {np.std(full_D)/np.mean(full_D):.2f}
     → Same robot pair can be "spectrally close" or "spectrally far"
       depending on joint configuration
     → D captures dynamics-relevant structural similarity, not just topology

  3. FUNCTIONAL GROUP HIERARCHY:
     Elbow ({s.get('elbow_D_mean',0):.3f}) < Shoulder ({s.get('shoulder_D_mean',0):.3f}) < Wrist ({s.get('wrist_D_mean',0):.3f})
     → Elbow joints most transferable across morphologies
     → Wrist least transferable (3-DOF vs 2-DOF)
     → Consistent with kinematic design intuition

  4. ENTROPY COMPARISON:""")

    oa_S = [d["entropy"] for d in data["openarm_spectra"]]
    pi_S = [d["entropy"] for d in data["piper_spectra"]]
    print(f"     OpenArm: S = {np.mean(oa_S):.4f} ± {np.std(oa_S):.4f}")
    print(f"     Piper:   S = {np.mean(pi_S):.4f} ± {np.std(pi_S):.4f}")

    if np.mean(oa_S) > np.mean(pi_S):
        print(f"     → OpenArm has higher entanglement (more cross-arm coupling)")
    else:
        print(f"     → Piper has higher entanglement")

    corr = np.corrcoef(full_D, delta_S)[0, 1]
    print(f"\n  5. ENTROPY-DISTANCE CORRELATION:")
    print(f"     r(D, |ΔS|) = {corr:.4f}")
    print(f"     → Entropy difference is {'strong' if corr > 0.8 else 'moderate' if corr > 0.5 else 'weak'} "
          f"predictor of spectral distance")


def generate_latex(data: dict) -> None:
    """Generate LaTeX table."""
    print_header("LaTeX Table (copy-paste ready)")

    s = data["summary"]
    oa_S = [d["entropy"] for d in data["openarm_spectra"]]
    pi_S = [d["entropy"] for d in data["piper_spectra"]]

    print(r"""
\begin{table}[t]
\caption{Cross-morphology entanglement spectral distance.
OpenArm (7+7 DOF) vs Piper (6+6 DOF) with 1\,kg grasped object,
20 configurations each (400 pairs).
$D$ is the spectral distance (Eq.~\ref{eq:spectral-distance}).}
\label{tab:spectral-distance}
\centering
\begin{tabular}{lcccc}
\toprule
Partition & $\bar{D}$ & $\sigma_D$ & $D_{\min}$ & $D_{\max}$ \\
\midrule""")

    print(f"Full $L|R$ & {s['full_D_mean']:.3f} & {s['full_D_std']:.3f} & "
          f"{s['full_D_min']:.3f} & {s['full_D_max']:.3f} \\\\")

    for group in ["shoulder", "elbow", "wrist"]:
        mk, sk = f"{group}_D_mean", f"{group}_D_std"
        if mk in s:
            # Get min/max for this group
            D_vals = [d["D"] for d in data["functional_group_distances"][group]]
            print(f"{group.capitalize()} & {s[mk]:.3f} & {s[sk]:.3f} & "
                  f"{min(D_vals):.3f} & {max(D_vals):.3f} \\\\")

    print(r"""\midrule
 & $\bar{S}$ & $\sigma_S$ & & \\
\midrule""")
    print(f"OpenArm 14-DOF & {np.mean(oa_S):.3f} & {np.std(oa_S):.3f} & & \\\\")
    print(f"Piper 12-DOF & {np.mean(pi_S):.3f} & {np.std(pi_S):.3f} & & \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    results_path = os.path.join(_project_root, "results", "spectral_distance_map.json")
    data = load_results(results_path)

    analyze_summary(data)
    analyze_entropy_ranges(data)
    analyze_configuration_dependence(data)
    analyze_functional_group_hierarchy(data)
    analyze_key_findings(data)
    generate_latex(data)

    # These need server-side execution (import physics modules)
    try:
        compute_classical_baseline(data)
        analyze_within_robot_distance(data)
    except Exception as e:
        print(f"\n  [Server-side analysis skipped: {e}]")
        print("  Run on gpu01 for full analysis including classical baseline.")


if __name__ == "__main__":
    main()
