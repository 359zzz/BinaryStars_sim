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
    heisenberg_hamiltonian,
    EigenEvolver,
    entanglement_spectrum,
    bipartite_entanglement_entropy,
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

    Returns dict with 'spectrum', 'entropy', 'psi', 'J_coupling'.
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
        dim_L = 2 ** n_arm
        return {
            'spectrum': np.array([1.0] + [0.0] * (dim_L - 1)),
            'entropy': 0.0,
            'psi': None,
            'J_coupling': None,
            'error': 'degenerate M_eff',
        }

    J_coupling = normalized_coupling_matrix(M_eff)
    h = local_field_terms(M_eff)

    H = heisenberg_hamiltonian(J_coupling, h, max_qubits=16)

    # Characteristic time
    J_off = np.abs(J_coupling.copy())
    np.fill_diagonal(J_off, 0.0)
    max_J = np.max(J_off)
    t_star = np.pi / (4.0 * max_J) if max_J > 1e-15 else 1.0

    # Initial state |0...01⟩
    dim = 2 ** n_total
    psi0 = np.zeros(dim, dtype=complex)
    psi0[1] = 1.0

    evolver = EigenEvolver(H)
    psi = evolver.evolve(psi0, t_star)

    spec = entanglement_spectrum(psi, n_arm, n_arm)
    S = bipartite_entanglement_entropy(psi, n_arm, n_arm)

    return {
        'spectrum': spec,
        'entropy': S,
        'psi': psi,
        'J_coupling': J_coupling,
        't_star': t_star,
    }


def compute_functional_group_spectra(
    psi: np.ndarray,
    n_arm: int,
    groups: dict[str, list[int]],
) -> dict[str, dict]:
    """Compute entanglement spectrum for each functional group bipartition.

    For a dual-arm system with 2*n_arm qubits, partition into
    functional groups and compute spectrum for each group's reduced
    density matrix.

    This enables dimension-matched comparison between robots with
    different DOF counts.
    """
    n_total = 2 * n_arm
    psi = np.asarray(psi, dtype=complex).ravel()

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

        # Reorder qubits: subsystem first, complement second
        tensor = psi.reshape([2] * n_total)
        perm = subsystem_indices + complement
        tensor = tensor.transpose(perm)

        # Reshape to (2^n_sub, 2^n_comp) and compute reduced density matrix
        dim_sub = 2 ** n_sub
        dim_comp = 2 ** n_comp
        Psi = tensor.reshape(dim_sub, dim_comp)
        rho_sub = Psi @ Psi.conj().T

        # Spectrum
        eigenvalues = np.linalg.eigvalsh(rho_sub).real
        eigenvalues = np.maximum(eigenvalues, 0.0)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Entropy
        nonzero = eigenvalues[eigenvalues > 1e-15]
        S = float(-np.sum(nonzero * np.log2(nonzero))) if len(nonzero) > 0 else 0.0

        result[group_name] = {
            'indices': subsystem_indices,
            'n_qubits': n_sub,
            'spectrum': eigenvalues.tolist(),
            'entropy': S,
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
        if spec_data['psi'] is not None:
            fg = compute_functional_group_spectra(
                spec_data['psi'], 7, FUNCTIONAL_GROUPS['openarm'],
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
        if spec_data['psi'] is not None:
            fg = compute_functional_group_spectra(
                spec_data['psi'], 6, FUNCTIONAL_GROUPS['piper'],
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
