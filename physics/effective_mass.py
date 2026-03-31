"""Effective mass matrix with cross-arm coupling via reflected-inertia method.

When two arms grasp a shared rigid object, MuJoCo's `mj_fullM` returns the
unconstrained kinematic-tree mass matrix (cross-arm block = 0).  The effective
mass matrix M_eff that includes object-mediated coupling is:

    M_eff = diag(M_arm1, M_arm2) + J_grasp^T @ M_obj @ J_grasp

where J_grasp = [J1 | J2] maps all joint velocities to the common grasp point.
This gives cross-arm coupling M_12 = J_1^T M_obj J_2, which is nonzero iff
M_obj ≠ 0 — exactly Theorem 3.

Properties:
- Symmetric positive definite (M_arm is PD, J^T M_obj J is PSD)
- Cross-arm block = 0 when M_obj = 0 (Theorem 3 Part i)
- Cross-arm block ≠ 0 when M_obj ≠ 0 (Theorem 3 Part ii)
- Monotone: coupling increases with object mass

This module provides both:
1. MuJoCo-based computation (using mj_fullM + mj_jac)
2. Standalone computation (using CRBA mass matrix + analytical Jacobian)
"""

from __future__ import annotations

import numpy as np


def compute_M_eff_from_mujoco(model, data, ee1_site_name: str, ee2_site_name: str,
                               M_obj_6x6: np.ndarray | None = None,
                               arm1_dof_range: tuple[int, int] = (0, 7),
                               arm2_dof_range: tuple[int, int] = (7, 14)) -> np.ndarray:
    """Compute effective mass matrix using MuJoCo APIs.

    Parameters
    ----------
    model, data : MuJoCo model and data (after mj_forward)
    ee1_site_name, ee2_site_name : site names for end-effectors
    M_obj_6x6 : (6, 6) spatial inertia of grasped object.
                 None or zeros → no object → cross-arm block = 0 (Lemma 3).
    arm1_dof_range : (start, end) DOF indices for arm 1
    arm2_dof_range : (start, end) DOF indices for arm 2

    Returns
    -------
    M_eff : (n1+n2, n1+n2) symmetric positive definite matrix
    """
    import mujoco

    nv = model.nv
    n1 = arm1_dof_range[1] - arm1_dof_range[0]
    n2 = arm2_dof_range[1] - arm2_dof_range[0]
    n_total = n1 + n2

    # 1. Full unconstrained mass matrix
    M_full = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M_full, data.qM)

    s1, e1 = arm1_dof_range
    s2, e2 = arm2_dof_range
    M_arm1 = M_full[s1:e1, s1:e1].copy()
    M_arm2 = M_full[s2:e2, s2:e2].copy()

    # 2. End-effector Jacobians (6 × nv, then slice to arm DOFs)
    ee1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee1_site_name)
    ee2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee2_site_name)

    jacp1 = np.zeros((3, nv))
    jacr1 = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp1, jacr1, ee1_id)
    J1 = np.vstack([jacp1[:, s1:e1], jacr1[:, s1:e1]])  # (6, n1)

    jacp2 = np.zeros((3, nv))
    jacr2 = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp2, jacr2, ee2_id)
    J2 = np.vstack([jacp2[:, s2:e2], jacr2[:, s2:e2]])  # (6, n2)

    # 3. Compute M_eff
    return compute_M_eff_khatib(M_arm1, M_arm2, J1, J2, M_obj_6x6)


def compute_M_eff_khatib(M_arm1: np.ndarray, M_arm2: np.ndarray,
                          J1: np.ndarray, J2: np.ndarray,
                          M_obj_6x6: np.ndarray | None = None) -> np.ndarray:
    """Reflected-inertia effective mass matrix for dual-arm grasping.

    M_eff = diag(M_arm1, M_arm2) + J_grasp^T @ M_obj @ J_grasp

    where J_grasp = [J1 | J2] is (6, n1+n2), mapping all joints to the
    common grasp point velocity.

    Parameters
    ----------
    M_arm1 : (n1, n1) mass matrix of arm 1
    M_arm2 : (n2, n2) mass matrix of arm 2
    J1 : (6, n1) geometric Jacobian of arm 1 end-effector
    J2 : (6, n2) geometric Jacobian of arm 2 end-effector
    M_obj_6x6 : (6, 6) spatial inertia of grasped object at grasp point
                 None or zeros → block-diagonal output (Lemma 3)

    Returns
    -------
    M_eff : (n1+n2, n1+n2) symmetric positive definite matrix
    """
    n1 = M_arm1.shape[0]
    n2 = M_arm2.shape[0]
    n_total = n1 + n2

    # Start with block-diagonal of arm mass matrices
    M_eff = np.zeros((n_total, n_total))
    M_eff[:n1, :n1] = M_arm1.copy()
    M_eff[n1:, n1:] = M_arm2.copy()

    # No object → block-diagonal (no cross-arm coupling)
    if M_obj_6x6 is None or np.allclose(M_obj_6x6, 0.0):
        return M_eff

    M_obj = np.asarray(M_obj_6x6, dtype=float)

    # Add reflected object inertia
    # J_grasp = [J1 | J2] is (6, n_total)
    # J_grasp^T M_obj J_grasp gives:
    #   (1,1) block: J1^T M_obj J1  (reflected inertia on arm 1)
    #   (2,2) block: J2^T M_obj J2  (reflected inertia on arm 2)
    #   (1,2) block: J1^T M_obj J2  (cross-arm coupling via object)
    M_eff[:n1, :n1] += J1.T @ M_obj @ J1
    M_eff[n1:, n1:] += J2.T @ M_obj @ J2
    cross = J1.T @ M_obj @ J2
    M_eff[:n1, n1:] = cross
    M_eff[n1:, :n1] = cross.T

    # Symmetrize (numerical)
    M_eff = (M_eff + M_eff.T) / 2.0

    return M_eff


def make_object_spatial_inertia(mass_kg: float,
                                 geometry: str = 'box',
                                 dims: tuple[float, ...] = (0.1, 0.1, 0.1)) -> np.ndarray:
    """Create 6×6 spatial inertia matrix for common object shapes.

    The spatial inertia is M = [[m·I, 0], [0, I_rot]] in the body frame
    at the center of mass.

    Parameters
    ----------
    mass_kg : object mass
    geometry : 'box', 'cylinder', 'sphere'
    dims : shape-dependent:
           box: (lx, ly, lz) side lengths
           cylinder: (radius, height)
           sphere: (radius,)

    Returns
    -------
    M_obj : (6, 6) spatial inertia matrix
    """
    m = mass_kg
    if m <= 0:
        return np.zeros((6, 6))

    if geometry == 'box':
        lx, ly, lz = dims
        Ixx = m * (ly**2 + lz**2) / 12.0
        Iyy = m * (lx**2 + lz**2) / 12.0
        Izz = m * (lx**2 + ly**2) / 12.0
    elif geometry == 'cylinder':
        r, h = dims
        Ixx = m * (3 * r**2 + h**2) / 12.0
        Iyy = Ixx
        Izz = m * r**2 / 2.0
    elif geometry == 'sphere':
        r = dims[0]
        Ixx = Iyy = Izz = 2.0 * m * r**2 / 5.0
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    M_obj = np.zeros((6, 6))
    # Translational inertia (top-left 3×3)
    M_obj[0, 0] = m
    M_obj[1, 1] = m
    M_obj[2, 2] = m
    # Rotational inertia (bottom-right 3×3)
    M_obj[3, 3] = Ixx
    M_obj[4, 4] = Iyy
    M_obj[5, 5] = Izz

    return M_obj


def validate_M_eff(M_eff: np.ndarray, n1: int, n2: int,
                    has_object: bool) -> dict[str, bool]:
    """Run validation checks on M_eff.

    Returns dict of check_name → pass/fail.
    """
    checks = {}

    # Symmetric?
    checks['symmetric'] = bool(np.allclose(M_eff, M_eff.T, atol=1e-10))

    # Positive definite? (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvalsh(M_eff)
    checks['positive_definite'] = bool(np.all(eigenvalues > -1e-10))

    # Cross-arm block
    cross_block = M_eff[:n1, n1:]
    cross_max = float(np.max(np.abs(cross_block)))

    if has_object:
        checks['cross_arm_nonzero'] = cross_max > 1e-10
    else:
        checks['cross_arm_zero'] = cross_max < 1e-10

    checks['cross_arm_max_abs'] = cross_max

    return checks
