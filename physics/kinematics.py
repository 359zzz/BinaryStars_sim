"""Forward kinematics and geometric Jacobian for serial chains.

Works with the parameter format from openarm_params.py / piper_params.py.
No MuJoCo dependency — pure numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physics.spatial import rotation_about_axis

FloatArray = NDArray[np.float64]


def forward_kinematics(
    n_joints: int,
    parent_indices: tuple[int, ...],
    joint_axes_local: np.ndarray,
    parent_to_joint_transforms: list[np.ndarray],
    q: np.ndarray,
) -> list[np.ndarray]:
    """Compute world-frame 4x4 transforms for all joints.

    Parameters
    ----------
    n_joints : number of joints
    parent_indices : parent index for each joint (-1 = world)
    joint_axes_local : (n_joints, 3) joint axes in local frame
    parent_to_joint_transforms : list of (4, 4) parent-to-joint transforms
    q : (n_joints,) joint configuration

    Returns
    -------
    T_world : list of (4, 4) world-frame transforms, one per joint
    """
    q = np.asarray(q, dtype=float)
    T_world: list[np.ndarray | None] = [None] * n_joints

    for i in range(n_joints):
        # Local transform: parent-to-joint * rotation(axis, q_i)
        T_pj = np.asarray(parent_to_joint_transforms[i], dtype=float)
        R_q = np.eye(4, dtype=float)
        R_q[:3, :3] = rotation_about_axis(joint_axes_local[i], float(q[i]))
        T_local = T_pj @ R_q

        parent = parent_indices[i]
        if parent < 0:
            T_world[i] = T_local
        else:
            T_world[i] = T_world[parent] @ T_local

    return T_world  # type: ignore[return-value]


def geometric_jacobian(
    n_joints: int,
    parent_indices: tuple[int, ...],
    joint_axes_local: np.ndarray,
    parent_to_joint_transforms: list[np.ndarray],
    q: np.ndarray,
    ee_offset: np.ndarray | None = None,
) -> tuple[FloatArray, np.ndarray]:
    """Compute 6 x n geometric Jacobian at end-effector.

    Parameters
    ----------
    n_joints, parent_indices, joint_axes_local, parent_to_joint_transforms :
        robot parameters (same as forward_kinematics)
    q : (n_joints,) joint configuration
    ee_offset : (3,) offset from last joint frame to EE point (default: zeros)

    Returns
    -------
    J : (6, n_joints) geometric Jacobian
        Rows 0-2: linear velocity (Jv = z_i × (p_ee - p_i))
        Rows 3-5: angular velocity (Jw = z_i)
    p_ee : (3,) EE position in world frame
    """
    T_world = forward_kinematics(
        n_joints, parent_indices, joint_axes_local,
        parent_to_joint_transforms, q,
    )

    # EE position
    T_last = T_world[-1]
    if ee_offset is not None:
        p_ee = T_last[:3, :3] @ np.asarray(ee_offset) + T_last[:3, 3]
    else:
        p_ee = T_last[:3, 3].copy()

    J = np.zeros((6, n_joints), dtype=float)

    for i in range(n_joints):
        T_i = T_world[i]
        R_i = T_i[:3, :3]
        p_i = T_i[:3, 3]

        # Joint axis in world frame
        z_i = R_i @ np.asarray(joint_axes_local[i], dtype=float)

        # Revolute joint: Jv = z × (p_ee - p_i), Jw = z
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i

    return J, p_ee


def compute_openarm_jacobian(q: np.ndarray) -> tuple[FloatArray, np.ndarray]:
    """Convenience: geometric Jacobian for OpenArm 7-DOF."""
    from physics.openarm_params import (
        N_JOINTS, PARENT_INDICES, JOINT_AXES,
        PARENT_TO_JOINT_TRANSFORMS, EE_OFFSET,
    )
    return geometric_jacobian(
        N_JOINTS, PARENT_INDICES, JOINT_AXES,
        PARENT_TO_JOINT_TRANSFORMS, q, EE_OFFSET,
    )


def compute_piper_jacobian(q: np.ndarray) -> tuple[FloatArray, np.ndarray]:
    """Convenience: geometric Jacobian for Piper 6-DOF."""
    from physics.piper_params import (
        N_JOINTS, PARENT_INDICES, JOINT_AXES,
        PARENT_TO_JOINT_TRANSFORMS, EE_OFFSET,
    )
    return geometric_jacobian(
        N_JOINTS, PARENT_INDICES, JOINT_AXES,
        PARENT_TO_JOINT_TRANSFORMS, q, EE_OFFSET,
    )
