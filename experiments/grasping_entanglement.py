"""S1.1: Grasping = Entanglement Generation (Theorem 3 verification).

Produces the core phase diagram: S(ρ_L) vs object mass for dual-arm grasping.

Key predictions:
- Part (i):  No object → S(ρ_L) = 0 (independent arms, product state)
- Part (ii): Object present → S(ρ_L) > 0 (entangled via object coupling)
- S(ρ_L) monotonically increases with object mass (stronger coupling)

Runs on OpenArm 14-DOF (7+7) and Piper 12-DOF (6+6) dual-arm systems.
Uses standalone CRBA + analytical Jacobian — no MuJoCo dependency.

Output: results/grasping_entanglement.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from physics.openarm_params import (
    compute_openarm_mass_matrix,
)
from physics.piper_params import (
    compute_piper_mass_matrix,
)
from physics.kinematics import (
    compute_openarm_jacobian,
    compute_piper_jacobian,
)
from physics.effective_mass import (
    compute_M_eff_khatib,
    make_object_spatial_inertia,
    validate_M_eff,
)
from quantum_prior.entanglement_graph import (
    normalized_coupling_matrix,
    local_field_terms,
    compute_entanglement_spectrum_from_mass_matrix,
    bipartite_entanglement_entropy,
    entanglement_spectrum,
)


# ── Experiment configurations ────────────────────────────────────────────────

OBJECT_MASSES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

OBJECT_GEOMETRY = 'box'
OBJECT_DIMS = (0.10, 0.10, 0.10)  # 10cm cube

# Representative arm configurations
CONFIGS = {
    'home': np.zeros(7),
    'elbow_bent': np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
    'reach_forward': np.array([0.0, 0.3, 0.0, 0.8, 0.0, 0.2, 0.0]),
}

PIPER_CONFIGS = {
    'home': np.zeros(6),
    'elbow_bent': np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0]),
    'reach_forward': np.array([0.0, 0.3, -0.5, 0.8, 0.0, 0.0]),
}


# ── Core computation ─────────────────────────────────────────────────────────

def run_single_condition(
    M_arm: np.ndarray,
    J_arm: np.ndarray,
    n_arm: int,
    object_mass: float,
) -> dict:
    """Run one (robot, config, mass) condition.

    Returns dict with entropy, spectrum, coupling info.
    """
    # Object spatial inertia
    if object_mass <= 0:
        M_obj = None
    else:
        M_obj = make_object_spatial_inertia(
            object_mass, OBJECT_GEOMETRY, OBJECT_DIMS,
        )

    # Effective mass matrix (Khatib)
    # Symmetric dual-arm: both arms at same config
    M_eff = compute_M_eff_khatib(M_arm, M_arm, J_arm, J_arm, M_obj)

    n_total = 2 * n_arm

    # Validate M_eff
    validation = validate_M_eff(M_eff, n_arm, n_arm, has_object=(object_mass > 0))

    # Cross-arm block analysis
    cross_block = M_eff[:n_arm, n_arm:]
    cross_max = float(np.max(np.abs(cross_block)))
    cross_mean = float(np.mean(np.abs(cross_block)))

    # Coupling matrix from M_eff
    diag = np.diag(M_eff)
    if np.any(diag <= 0):
        # M_eff degenerate — skip entanglement computation
        return {
            'object_mass': object_mass,
            'entropy': 0.0,
            'spectrum': [],
            'cross_arm_max': cross_max,
            'cross_arm_mean': cross_mean,
            'validation': {k: v for k, v in validation.items()
                          if not isinstance(v, np.ndarray)},
            'error': 'M_eff has non-positive diagonal',
        }

    J_coupling = normalized_coupling_matrix(M_eff)
    h_fields = local_field_terms(M_eff)

    # Cross-arm coupling from J_coupling
    J_cross = J_coupling[:n_arm, n_arm:]
    j_cross_max = float(np.max(np.abs(J_cross)))
    j_cross_mean = float(np.mean(np.abs(J_cross)))

    # Entanglement spectrum computation
    result = compute_entanglement_spectrum_from_mass_matrix(
        M_eff, n_L=n_arm, n_R=n_arm,
    )

    return {
        'object_mass': object_mass,
        'entropy': result['entropy'],
        'spectrum': result['spectrum'].tolist(),
        't_star': result['t_star'],
        'cross_arm_max_M': cross_max,
        'cross_arm_mean_M': cross_mean,
        'cross_arm_max_J': j_cross_max,
        'cross_arm_mean_J': j_cross_mean,
        'validation': {k: v for k, v in validation.items()
                      if not isinstance(v, np.ndarray)},
    }


def run_robot_experiment(
    robot_name: str,
    n_arm: int,
    compute_mass: callable,
    compute_jacobian: callable,
    configs: dict[str, np.ndarray],
) -> dict:
    """Run full experiment for one robot type."""
    robot_results = {
        'robot_name': robot_name,
        'n_arm': n_arm,
        'n_total': 2 * n_arm,
        'configs': {},
    }

    for config_name, q in configs.items():
        print(f"\n  Config: {config_name} (q = {q.tolist()})")

        # Compute single-arm mass matrix and Jacobian
        M_arm = compute_mass(q)
        J_arm, p_ee = compute_jacobian(q)

        print(f"    M_arm condition: {np.linalg.cond(M_arm):.1f}")
        print(f"    J_arm rank: {np.linalg.matrix_rank(J_arm, tol=1e-6)}")
        print(f"    EE position: {p_ee}")

        config_results = {
            'q': q.tolist(),
            'ee_position': p_ee.tolist(),
            'M_arm_diag': np.diag(M_arm).tolist(),
            'conditions': [],
        }

        for mass in OBJECT_MASSES:
            t0 = time.time()
            result = run_single_condition(M_arm, J_arm, n_arm, mass)
            dt = time.time() - t0

            result['compute_time_s'] = dt
            config_results['conditions'].append(result)

            S = result['entropy']
            cross_J = result.get('cross_arm_max_J', 0.0)
            tag = "PASS" if (mass == 0 and S < 1e-10) or (mass > 0 and S > 0) else "FAIL"
            print(f"    mass={mass:.1f} kg: S(ρ_L)={S:.6f}, "
                  f"|J_cross|_max={cross_J:.4f}, "
                  f"t={dt:.3f}s [{tag}]")

        robot_results['configs'][config_name] = config_results

    return robot_results


# ── Theorem 3 verification ───────────────────────────────────────────────────

def verify_theorem3(results: dict) -> dict:
    """Check Theorem 3 predictions across all conditions."""
    checks = {
        'part_i_zero_entropy': [],   # S=0 when no object
        'part_ii_nonzero_entropy': [],  # S>0 when object present
        'monotonicity': [],          # S increases with mass
    }

    for robot_name, robot_data in results['robots'].items():
        for config_name, config_data in robot_data['configs'].items():
            conditions = config_data['conditions']

            # Part (i): no object → S = 0
            no_obj = [c for c in conditions if c['object_mass'] == 0.0]
            for c in no_obj:
                passed = c['entropy'] < 1e-10
                checks['part_i_zero_entropy'].append({
                    'robot': robot_name,
                    'config': config_name,
                    'S': c['entropy'],
                    'passed': passed,
                })

            # Part (ii): object → S > 0
            with_obj = [c for c in conditions if c['object_mass'] > 0.0]
            for c in with_obj:
                passed = c['entropy'] > 1e-10
                checks['part_ii_nonzero_entropy'].append({
                    'robot': robot_name,
                    'config': config_name,
                    'mass': c['object_mass'],
                    'S': c['entropy'],
                    'passed': passed,
                })

            # Monotonicity: S(m1) ≤ S(m2) for m1 < m2
            entropies = [(c['object_mass'], c['entropy']) for c in conditions]
            entropies.sort(key=lambda x: x[0])
            mono = True
            for k in range(1, len(entropies)):
                if entropies[k][1] < entropies[k-1][1] - 1e-10:
                    mono = False
                    break
            checks['monotonicity'].append({
                'robot': robot_name,
                'config': config_name,
                'passed': mono,
                'entropies': [(m, s) for m, s in entropies],
            })

    # Summary
    n_i = len(checks['part_i_zero_entropy'])
    n_i_pass = sum(1 for c in checks['part_i_zero_entropy'] if c['passed'])
    n_ii = len(checks['part_ii_nonzero_entropy'])
    n_ii_pass = sum(1 for c in checks['part_ii_nonzero_entropy'] if c['passed'])
    n_mono = len(checks['monotonicity'])
    n_mono_pass = sum(1 for c in checks['monotonicity'] if c['passed'])

    checks['summary'] = {
        'part_i': f"{n_i_pass}/{n_i}",
        'part_ii': f"{n_ii_pass}/{n_ii}",
        'monotonicity': f"{n_mono_pass}/{n_mono}",
        'all_passed': (n_i_pass == n_i and n_ii_pass == n_ii and n_mono_pass == n_mono),
    }

    return checks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("S1.1: Grasping = Entanglement Generation (Theorem 3)")
    print("=" * 70)

    results = {
        'experiment': 'grasping_entanglement',
        'object_geometry': OBJECT_GEOMETRY,
        'object_dims': OBJECT_DIMS,
        'object_masses': OBJECT_MASSES,
        'robots': {},
    }

    # OpenArm 14-DOF (7+7) dual-arm
    print("\n" + "─" * 50)
    print("Robot: OpenArm 14-DOF (7+7)")
    print("─" * 50)
    results['robots']['openarm'] = run_robot_experiment(
        'OpenArm', 7,
        compute_openarm_mass_matrix,
        compute_openarm_jacobian,
        CONFIGS,
    )

    # Piper 12-DOF (6+6) dual-arm
    print("\n" + "─" * 50)
    print("Robot: Piper 12-DOF (6+6)")
    print("─" * 50)
    results['robots']['piper'] = run_robot_experiment(
        'Piper', 6,
        compute_piper_mass_matrix,
        compute_piper_jacobian,
        PIPER_CONFIGS,
    )

    # Theorem 3 verification
    print("\n" + "=" * 70)
    print("Theorem 3 Verification")
    print("=" * 70)
    theorem3 = verify_theorem3(results)
    results['theorem3_verification'] = theorem3

    print(f"\n  Part (i)  — S=0 when no object:  {theorem3['summary']['part_i']}")
    print(f"  Part (ii) — S>0 when object:      {theorem3['summary']['part_ii']}")
    print(f"  Monotonicity — S↑ with mass:       {theorem3['summary']['monotonicity']}")
    print(f"\n  ALL PASSED: {theorem3['summary']['all_passed']}")

    # Save results
    results_dir = os.path.join(_project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'grasping_entanglement.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
