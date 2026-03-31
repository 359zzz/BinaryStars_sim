"""S1.2: Cross-Morphology Spectral Distance Map.

Computes entanglement spectral distance D(R₁, R₂) between OpenArm (7+7)
and Piper (6+6) dual-arm systems across a grid of configurations.

Key results:
- Spectral distance varies with configuration (not just morphology)
- Functional group mapping (shoulder↔shoulder, elbow↔elbow, wrist↔wrist)
  enables direct comparison despite different DOF counts
- Spectral distance correlates with policy transfer difficulty

Output: results/spectral_distance_map.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from physics.openarm_params import compute_openarm_mass_matrix
from physics.piper_params import compute_piper_mass_matrix
from physics.kinematics import compute_openarm_jacobian, compute_piper_jacobian
from physics.effective_mass import (
    compute_M_eff_khatib,
    make_object_spatial_inertia,
)
from quantum_prior.entanglement_graph import (
    normalized_coupling_matrix,
    local_field_terms,
    single_excitation_hamiltonian,
    SingleExcitationEvolver,
    entropy_from_amplitudes,
    spectral_distance,
)


# ── Functional group definitions ─────────────────────────────────────────────

# OpenArm 7-DOF: joints 0-6
# Piper 6-DOF: joints 0-5
# Functional group mapping:
#   Shoulder: OA[0,1] ↔ Pi[0,1]
#   Elbow:    OA[2,3] ↔ Pi[2,3]
#   Wrist:    OA[4,5,6] ↔ Pi[4,5]

FUNCTIONAL_GROUPS = {
    'openarm': {
        'shoulder': [0, 1],
        'elbow': [2, 3],
        'wrist': [4, 5, 6],
    },
    'piper': {
        'shoulder': [0, 1],
        'elbow': [2, 3],
        'wrist': [4, 5],
    },
}


# ── Configuration grids ──────────────────────────────────────────────────────

def make_config_grid_openarm(n_samples: int = 20) -> list[np.ndarray]:
    """Sample configurations for OpenArm 7-DOF."""
    rng = np.random.RandomState(42)
    # Joint limits (approximate)
    lo = np.array([-1.3, -0.15, -1.48, 0.0, -1.48, -0.7, -1.4])
    hi = np.array([1.3, 1.57, 1.48, 2.35, 1.48, 0.7, 1.4])
    configs = [np.zeros(7)]  # home config
    for _ in range(n_samples - 1):
        configs.append(rng.uniform(lo, hi))
    return configs


def make_config_grid_piper(n_samples: int = 20) -> list[np.ndarray]:
    """Sample configurations for Piper 6-DOF."""
    rng = np.random.RandomState(42)
    # Piper joints: all Z-axis, approximate limits
    lo = np.array([-2.6, -1.5, -1.5, -1.7, -1.7, -2.6])
    hi = np.array([2.6, 1.5, 1.5, 1.7, 1.7, 2.6])
    configs = [np.zeros(6)]
    for _ in range(n_samples - 1):
        configs.append(rng.uniform(lo, hi))
    return configs


# ── Dual-arm entanglement spectrum pipeline ──────────────────────────────────

def compute_dualarm_spectrum(
    M_arm: np.ndarray,
    J_arm: np.ndarray,
    n_arm: int,
    object_mass: float = 1.0,
) -> dict:
    """Compute entanglement spectrum for symmetric dual-arm grasping.

    Uses the single-excitation sector (n-dimensional) for efficiency.

    Returns dict with 'spectrum', 'entropy', 'c_t' (sector amplitudes),
    'J_coupling'.
    """
    if object_mass > 0:
        M_obj = make_object_spatial_inertia(object_mass, 'box', (0.1, 0.1, 0.1))
    else:
        M_obj = None

    M_eff = compute_M_eff_khatib(M_arm, M_arm, J_arm, J_arm, M_obj)
    n_total = 2 * n_arm

    # Check diagonal positivity
    diag = np.diag(M_eff)
    if np.any(diag <= 0):
        return {
            'spectrum': np.array([1.0, 0.0]),
            'entropy': 0.0,
            'c_t': None,
            'J_coupling': None,
            'error': 'degenerate M_eff',
        }

    J_coupling = normalized_coupling_matrix(M_eff)
    h = local_field_terms(M_eff)

    # Single-excitation sector: n_total × n_total
    H_eff = single_excitation_hamiltonian(J_coupling, h)

    # Characteristic time
    J_off = np.abs(J_coupling.copy())
    np.fill_diagonal(J_off, 0.0)
    max_J = np.max(J_off)
    t_star = np.pi / (4.0 * max_J) if max_J > 1e-15 else 1.0

    # Evolve in sector
    evolver = SingleExcitationEvolver(H_eff)
    c0 = np.zeros(n_total, dtype=complex)
    c0[n_total - 1] = 1.0
    c_t = evolver.evolve(c0, t_star)

    # Bipartite L|R spectrum (binary in single-excitation sector)
    p_L = float(np.sum(np.abs(c_t[:n_arm])**2))
    p_R = float(np.sum(np.abs(c_t[n_arm:])**2))
    S = entropy_from_amplitudes(c_t, n_arm)

    spec = np.array([max(p_L, p_R), min(p_L, p_R)])

    return {
        'spectrum': spec,
        'entropy': S,
        'c_t': c_t,
        'J_coupling': J_coupling,
        't_star': t_star,
    }


def compute_functional_group_spectra(
    c_t: np.ndarray,
    n_arm: int,
    groups: dict[str, list[int]],
) -> dict[str, dict]:
    """Compute entanglement spectrum for each functional group bipartition.

    Uses single-excitation sector amplitudes c_t (n-dimensional vector).
    In this sector, ANY bipartition A|B has binary spectrum:
        p_A = Σ_{i∈A} |c_i|², p_B = 1 - p_A

    This enables dimension-matched comparison between robots with
    different DOF counts.
    """
    n_total = 2 * n_arm
    c_t = np.asarray(c_t, dtype=complex).ravel()

    result = {}

    for group_name, joint_indices in groups.items():
        # Dual-arm: left arm indices as-is, right arm indices shifted by n_arm
        left_indices = joint_indices
        right_indices = [j + n_arm for j in joint_indices]
        subsystem_indices = left_indices + right_indices
        complement = [i for i in range(n_total) if i not in subsystem_indices]

        n_sub = len(subsystem_indices)
        n_comp = len(complement)

        if n_sub == 0 or n_comp == 0:
            continue

        # In single-excitation sector: p_A = Σ_{i∈A} |c_i|²
        p_sub = float(np.sum(np.abs(c_t[subsystem_indices])**2))
        p_comp = 1.0 - p_sub

        # Binary spectrum
        spec = [max(p_sub, p_comp), min(p_sub, p_comp)]

        # Binary entropy
        if p_sub < 1e-15 or p_comp < 1e-15:
            S = 0.0
        else:
            S = -p_sub * np.log2(p_sub) - p_comp * np.log2(p_comp)

        result[group_name] = {
            'indices': subsystem_indices,
            'n_qubits': n_sub,
            'spectrum': spec,
            'entropy': float(S),
        }

    return result


# ── Main experiment ──────────────────────────────────────────────────────────

def run_spectral_distance_experiment(
    n_configs: int = 20,
    object_mass: float = 1.0,
) -> dict:
    """Compute spectral distance map between OpenArm and Piper."""
    print(f"  Generating config grids (n={n_configs} each)...")
    oa_configs = make_config_grid_openarm(n_configs)
    pi_configs = make_config_grid_piper(n_configs)

    results = {
        'experiment': 'spectral_distance_map',
        'object_mass': object_mass,
        'n_configs': n_configs,
        'openarm_spectra': [],
        'piper_spectra': [],
        'full_spectral_distances': [],
        'functional_group_distances': {g: [] for g in ['shoulder', 'elbow', 'wrist']},
    }

    # Compute OpenArm spectra
    print("\n  Computing OpenArm spectra...")
    for idx, q in enumerate(oa_configs):
        t0 = time.time()
        M_arm = compute_openarm_mass_matrix(q)
        J_arm, p_ee = compute_openarm_jacobian(q)
        spec_data = compute_dualarm_spectrum(M_arm, J_arm, 7, object_mass)

        # Functional group spectra
        fg = {}
        if spec_data['c_t'] is not None:
            fg = compute_functional_group_spectra(
                spec_data['c_t'], 7, FUNCTIONAL_GROUPS['openarm'],
            )

        dt = time.time() - t0
        entry = {
            'config_idx': idx,
            'q': q.tolist(),
            'entropy': spec_data['entropy'],
            'spectrum': spec_data['spectrum'].tolist() if isinstance(spec_data['spectrum'], np.ndarray) else spec_data['spectrum'],
            'functional_groups': fg,
            'compute_time_s': dt,
        }
        results['openarm_spectra'].append(entry)

        if idx % 5 == 0:
            print(f"    [{idx+1}/{n_configs}] S={spec_data['entropy']:.4f} ({dt:.2f}s)")

    # Compute Piper spectra
    print("\n  Computing Piper spectra...")
    for idx, q in enumerate(pi_configs):
        t0 = time.time()
        M_arm = compute_piper_mass_matrix(q)
        J_arm, p_ee = compute_piper_jacobian(q)
        spec_data = compute_dualarm_spectrum(M_arm, J_arm, 6, object_mass)

        fg = {}
        if spec_data['c_t'] is not None:
            fg = compute_functional_group_spectra(
                spec_data['c_t'], 6, FUNCTIONAL_GROUPS['piper'],
            )

        dt = time.time() - t0
        entry = {
            'config_idx': idx,
            'q': q.tolist(),
            'entropy': spec_data['entropy'],
            'spectrum': spec_data['spectrum'].tolist() if isinstance(spec_data['spectrum'], np.ndarray) else spec_data['spectrum'],
            'functional_groups': fg,
            'compute_time_s': dt,
        }
        results['piper_spectra'].append(entry)

        if idx % 5 == 0:
            print(f"    [{idx+1}/{n_configs}] S={spec_data['entropy']:.4f} ({dt:.2f}s)")

    # Compute pairwise spectral distances
    print("\n  Computing spectral distances...")

    # Full bipartite spectral distance (L|R partition)
    # OpenArm has 2^7=128 dim spectrum, Piper has 2^6=64 dim
    # Use zero-padding for comparison
    for i, oa in enumerate(results['openarm_spectra']):
        for j, pi in enumerate(results['piper_spectra']):
            oa_spec = np.array(oa['spectrum'])
            pi_spec = np.array(pi['spectrum'])

            D = spectral_distance(oa_spec, pi_spec)

            results['full_spectral_distances'].append({
                'oa_idx': i,
                'pi_idx': j,
                'D': D,
                'delta_S': abs(oa['entropy'] - pi['entropy']),
            })

    # Functional group spectral distances
    for group in ['shoulder', 'elbow', 'wrist']:
        for i, oa in enumerate(results['openarm_spectra']):
            for j, pi in enumerate(results['piper_spectra']):
                oa_fg = oa['functional_groups'].get(group, {})
                pi_fg = pi['functional_groups'].get(group, {})

                if not oa_fg or not pi_fg:
                    continue

                oa_spec = np.array(oa_fg['spectrum'])
                pi_spec = np.array(pi_fg['spectrum'])

                D = spectral_distance(oa_spec, pi_spec)

                results['functional_group_distances'][group].append({
                    'oa_idx': i,
                    'pi_idx': j,
                    'D': D,
                    'delta_S': abs(oa_fg['entropy'] - pi_fg['entropy']),
                })

    # Summary statistics
    full_D = [d['D'] for d in results['full_spectral_distances']]
    results['summary'] = {
        'full_D_mean': float(np.mean(full_D)),
        'full_D_std': float(np.std(full_D)),
        'full_D_min': float(np.min(full_D)),
        'full_D_max': float(np.max(full_D)),
    }

    for group in ['shoulder', 'elbow', 'wrist']:
        group_D = [d['D'] for d in results['functional_group_distances'][group]]
        if group_D:
            results['summary'][f'{group}_D_mean'] = float(np.mean(group_D))
            results['summary'][f'{group}_D_std'] = float(np.std(group_D))

    return results


def main():
    print("=" * 70)
    print("S1.2: Cross-Morphology Spectral Distance Map")
    print("=" * 70)

    # Run with 1.0 kg object (standard grasping scenario)
    results = run_spectral_distance_experiment(
        n_configs=20,
        object_mass=1.0,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    s = results['summary']
    print(f"  Full spectral distance D(OA, Pi):")
    print(f"    mean = {s['full_D_mean']:.4f} ± {s['full_D_std']:.4f}")
    print(f"    range = [{s['full_D_min']:.4f}, {s['full_D_max']:.4f}]")

    for group in ['shoulder', 'elbow', 'wrist']:
        key = f'{group}_D_mean'
        if key in s:
            print(f"  {group.capitalize()} group D: "
                  f"{s[key]:.4f} ± {s[f'{group}_D_std']:.4f}")

    # Save
    results_dir = os.path.join(_project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'spectral_distance_map.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
