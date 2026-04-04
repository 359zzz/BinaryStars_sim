"""Dual-arm effective mass matrix for quantum entanglement computation.

Computes M_eff(q) = diag(M_L, M_R) + J_grasp^T @ M_obj @ J_grasp
for the 14-DOF OpenArm dual-arm system with a virtual shared object.

Used by CachedEntanglementComputer to compute entanglement graph C_ij.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from physics.effective_mass import compute_M_eff_khatib, make_object_spatial_inertia
from physics.kinematics import compute_openarm_jacobian
from physics.openarm_params import compute_openarm_mass_matrix


def compute_dualarm_mass_matrix(
    q: np.ndarray,
    object_mass: float = 1.0,
    object_geometry: str = "box",
    object_dims: tuple[float, ...] = (0.1, 0.1, 0.1),
) -> np.ndarray:
    """Compute 14x14 effective mass matrix for dual-arm with virtual object.

    Parameters
    ----------
    q : (14,) joint angles [q_L(7), q_R(7)]
    object_mass : mass of virtual shared object (kg). 0 = no cross-arm coupling.
    object_geometry : shape for inertia computation
    object_dims : dimensions for inertia computation

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

    # Object spatial inertia (6x6)
    M_obj = make_object_spatial_inertia(object_mass, object_geometry, object_dims)

    # Assemble M_eff (14x14)
    return compute_M_eff_khatib(M_L, M_R, J_L, J_R, M_obj)


def make_dualarm_mass_fn(
    object_mass: float = 1.0,
    object_geometry: str = "box",
    object_dims: tuple[float, ...] = (0.1, 0.1, 0.1),
):
    """Create a mass_matrix_fn compatible with CachedEntanglementComputer.

    Usage:
        mass_fn = make_dualarm_mass_fn(object_mass=1.0)
        qc = CachedEntanglementComputer(mass_matrix_fn=mass_fn, ...)
    """
    return partial(
        compute_dualarm_mass_matrix,
        object_mass=object_mass,
        object_geometry=object_geometry,
        object_dims=object_dims,
    )
