"""Dual-arm effective mass matrix for quantum entanglement computation.

Computes M_eff(q) = diag(M_L, M_R) + J_grasp^T @ M_obj @ J_grasp
for the 14-DOF OpenArm dual-arm system with a shared aluminum profile.

Object: 4040 aluminum profile, 0.4m x 0.04m x 0.04m, ~0.5 kg.
Each arm grasps one end of the profile.

Used by CachedEntanglementComputer to compute entanglement graph C_ij.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from physics.effective_mass import compute_M_eff_khatib, make_object_spatial_inertia
from physics.kinematics import compute_openarm_jacobian
from physics.openarm_params import compute_openarm_mass_matrix

# 4040 aluminum profile defaults (consistent with openarm14_dual.xml)
PROFILE_LENGTH = 0.4    # m
PROFILE_SECTION = 0.04  # m (40mm x 40mm cross-section)
PROFILE_MASS = 0.5      # kg


def compute_dualarm_mass_matrix(
    q: np.ndarray,
    object_mass: float = PROFILE_MASS,
    object_geometry: str = "box",
    object_dims: tuple[float, ...] = (PROFILE_LENGTH, PROFILE_SECTION, PROFILE_SECTION),
) -> np.ndarray:
    """Compute 14x14 effective mass matrix for dual-arm with shared object.

    Parameters
    ----------
    q : (14,) joint angles [q_L(7), q_R(7)]
    object_mass : mass of shared object (kg). 0 = no cross-arm coupling.
    object_geometry : 'box', 'cylinder', 'sphere'
    object_dims : dimensions matching geometry type.
                  Default: (0.4, 0.04, 0.04) = 4040 aluminum profile.

    Returns
    -------
    M_eff : (14, 14) symmetric positive definite mass matrix
    """
    q = np.asarray(q, dtype=np.float64)
    q_L, q_R = q[:7], q[7:]

    # Single-arm mass matrices (7x7 each)
    M_L = compute_openarm_mass_matrix(q_L)
    M_R = compute_openarm_mass_matrix(q_R)

    # Single-arm Jacobians at EE (6x7 each)
    J_L, _ = compute_openarm_jacobian(q_L)
    J_R, _ = compute_openarm_jacobian(q_R)

    # Object spatial inertia (6x6) — aluminum profile
    M_obj = make_object_spatial_inertia(object_mass, object_geometry, object_dims)

    # Assemble M_eff (14x14)
    return compute_M_eff_khatib(M_L, M_R, J_L, J_R, M_obj)


def make_dualarm_mass_fn(
    object_mass: float = PROFILE_MASS,
    object_geometry: str = "box",
    object_dims: tuple[float, ...] = (PROFILE_LENGTH, PROFILE_SECTION, PROFILE_SECTION),
):
    """Create a mass_matrix_fn compatible with CachedEntanglementComputer.

    Default: 4040 aluminum profile (0.4m x 0.04m x 0.04m, 0.5 kg).
    Anisotropic inertia: I_xx=1.33e-4 << I_yy=I_zz=6.73e-3.

    Usage:
        mass_fn = make_dualarm_mass_fn(object_mass=0.5)
        qc = CachedEntanglementComputer(mass_matrix_fn=mass_fn, ...)
    """
    return partial(
        compute_dualarm_mass_matrix,
        object_mass=object_mass,
        object_geometry=object_geometry,
        object_dims=object_dims,
    )
