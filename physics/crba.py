"""Composite Rigid Body Algorithm (CRBA) for mass matrix computation.

Standalone port — depends only on spatial.py and numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physics.spatial import (
    spatial_inertia,
    spatial_transform_inverse,
    rotation_about_axis,
)

FloatArray = NDArray[np.float64]


def compute_mass_matrix(
    n_joints: int,
    parent_indices: tuple[int, ...],
    joint_axes_local: np.ndarray,
    parent_to_joint_transforms: list[np.ndarray] | tuple[np.ndarray, ...],
    link_masses: np.ndarray,
    link_inertias: np.ndarray,
    link_com_local: np.ndarray,
    q: np.ndarray,
) -> FloatArray:
    """CRBA: compute n x n symmetric positive-definite mass matrix M(q).

    Parameters
    ----------
    n_joints : number of joints
    parent_indices : per-joint parent index (-1 = root)
    joint_axes_local : (n, 3) local rotation axes
    parent_to_joint_transforms : n x (4, 4) homogeneous transforms
    link_masses : (n,) link masses in kg
    link_inertias : (n, 3, 3) inertia tensors about CoM
    link_com_local : (n, 3) CoM in link frame
    q : (n,) joint angles in rad
    """
    n = n_joints
    q = np.asarray(q, dtype=float)

    # Forward pass
    X_lambda: list[FloatArray] = []
    Ic: list[FloatArray] = []
    S: list[FloatArray] = []

    for i in range(n):
        axis_local = joint_axes_local[i]
        R_joint = rotation_about_axis(axis_local, q[i])

        T_pj = parent_to_joint_transforms[i]
        R_pj = T_pj[:3, :3]
        p_pj = T_pj[:3, 3]

        R_total = R_pj @ R_joint
        X_i = spatial_transform_inverse(R_total.T, p_pj)
        X_lambda.append(X_i)

        Ic_i = spatial_inertia(
            float(link_masses[i]),
            link_inertias[i],
            link_com_local[i],
        )
        Ic.append(Ic_i)

        s_i = np.zeros(6, dtype=float)
        s_i[:3] = axis_local
        S.append(s_i)

    # Backward pass: accumulate composite inertias
    for i in range(n - 1, -1, -1):
        parent = parent_indices[i]
        if parent >= 0:
            Ic[parent] = Ic[parent] + X_lambda[i].T @ Ic[i] @ X_lambda[i]

    # Mass matrix assembly
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        F = Ic[i] @ S[i]
        M[i, i] = float(S[i] @ F)

        j = i
        while parent_indices[j] >= 0:
            F = X_lambda[j].T @ F
            j = parent_indices[j]
            M[i, j] = float(S[j] @ F)
            M[j, i] = M[i, j]

    return M
