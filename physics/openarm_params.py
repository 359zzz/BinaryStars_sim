"""Hardcoded OpenArm v10 7-DOF parameters for standalone CRBA.

Sourced from binarystars openarm_data.py — no DynamicsIR dependency.
"""

from __future__ import annotations

import math

import numpy as np

from physics.spatial import rpy_to_rotation

# ── Link parameters (from inertials.yaml) ──────────────────────────────────

LINK_MASSES = np.array([
    1.1416684646202298,  # link1
    0.2775092746011571,  # link2
    1.073863338202347,   # link3
    0.6348534566833373,  # link4
    0.6156588026168502,  # link5
    0.475202773187987,   # link6
    0.4659771327380578,  # link7
], dtype=float)

LINK_COMS = np.array([
    [0.0011467657911800769, 3.319987657026362e-05, 0.05395284380736254],
    [0.00839629182351943, -2.0145102027597523e-08, 0.03256649300522363],
    [-0.002104752099628911, 0.0005549085042607548, 0.09047470545721961],
    [-0.0029006831074562967, -0.03030575826634669, 0.06339637422196209],
    [-0.003049665024221911, 0.0008866902457326625, 0.043079803024980934],
    [-0.037136587005447405, 0.00033230528343419053, -9.498374522309838e-05],
    [6.875510271106056e-05, 0.01266175250761268, 0.06951945409987448],
], dtype=float)

LINK_INERTIAS = np.array([
    [[0.001567, -1e-06, -2.9e-05], [-1e-06, 0.001273, 1e-06], [-2.9e-05, 1e-06, 0.001016]],
    [[0.000359, 1e-06, -0.000109], [1e-06, 0.000376, 1e-06], [-0.000109, 1e-06, 0.000232]],
    [[0.004372, 1e-06, 1.1e-05], [1e-06, 0.004319, -3.6e-05], [1.1e-05, -3.6e-05, 0.000661]],
    [[0.000623, -1e-06, -1.9e-05], [-1e-06, 0.000511, 3.8e-05], [-1.9e-05, 3.8e-05, 0.000334]],
    [[0.000423, -8e-06, 6e-06], [-8e-06, 0.000445, -6e-06], [6e-06, -6e-06, 0.000324]],
    [[0.000143, 1e-06, 1e-06], [1e-06, 0.000157, 1e-06], [1e-06, 1e-06, 0.000159]],
    [[0.000639, 1e-06, 1e-06], [1e-06, 0.000497, 8.9e-05], [1e-06, 8.9e-05, 0.000342]],
], dtype=float)

# ── Joint parameters ────────────────────────────────────────────────────────

JOINT_AXES = np.array([
    [0.0, 0.0, 1.0],   # j1: Z
    [-1.0, 0.0, 0.0],  # j2: -X
    [0.0, 0.0, 1.0],   # j3: Z
    [0.0, 1.0, 0.0],   # j4: Y
    [0.0, 0.0, 1.0],   # j5: Z
    [1.0, 0.0, 0.0],   # j6: X
    [0.0, 1.0, 0.0],   # j7: Y
], dtype=float)

JOINT_XYZ = [
    (0.0, 0.0, 0.0625),
    (-0.0301, 0.0, 0.06),
    (0.0301, 0.0, 0.06625),
    (0.0, 0.0315, 0.15375),
    (0.0, -0.0315, 0.0955),
    (0.0375, 0.0, 0.1205),
    (-0.0375, 0.0, 0.0),
]

JOINT_RPY = [
    (0.0, 0.0, 0.0),
    (math.pi / 2, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
]

PARENT_INDICES = (-1, 0, 1, 2, 3, 4, 5)

N_JOINTS = 7

# EE offset from last joint frame
EE_OFFSET = np.array([0.0, 0.0, 0.07], dtype=float)


def build_transforms() -> list[np.ndarray]:
    """Build parent-to-joint 4x4 homogeneous transforms."""
    transforms = []
    for i in range(N_JOINTS):
        T = np.eye(4, dtype=float)
        T[:3, :3] = rpy_to_rotation(*JOINT_RPY[i])
        T[:3, 3] = JOINT_XYZ[i]
        transforms.append(T)
    return transforms


# Pre-built transforms (module-level constant)
PARENT_TO_JOINT_TRANSFORMS = build_transforms()


def compute_openarm_mass_matrix(q: np.ndarray) -> np.ndarray:
    """Convenience: compute M(q) for the standard OpenArm 7-DOF."""
    from physics.crba import compute_mass_matrix
    return compute_mass_matrix(
        n_joints=N_JOINTS,
        parent_indices=PARENT_INDICES,
        joint_axes_local=JOINT_AXES,
        parent_to_joint_transforms=PARENT_TO_JOINT_TRANSFORMS,
        link_masses=LINK_MASSES,
        link_inertias=LINK_INERTIAS,
        link_com_local=LINK_COMS,
        q=q,
    )


def compute_openarm_coupling(q: np.ndarray) -> np.ndarray:
    """Convenience: compute J_ij(q) coupling matrix for OpenArm."""
    from physics.coupling import normalized_coupling_matrix
    M = compute_openarm_mass_matrix(q)
    return normalized_coupling_matrix(M)


def modify_payload(payload_kg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return modified (masses, inertias, coms) with payload at EE.

    Payload m_p attached at EE offset r=(0,0,0.07) from link7 frame.
    """
    masses = LINK_MASSES.copy()
    inertias = LINK_INERTIAS.copy()
    coms = LINK_COMS.copy()

    m6 = masses[6]
    c6 = coms[6].copy()
    r = EE_OFFSET

    # Update mass
    masses[6] = m6 + payload_kg

    # Update CoM
    coms[6] = (m6 * c6 + payload_kg * r) / (m6 + payload_kg)

    # Update inertia: point mass contribution at r relative to new CoM
    # Use parallel axis theorem about the LINK CoM
    # Delta I from point mass at r: m_p * (r^T r I - r r^T)
    inertias[6] = inertias[6].copy()
    dI = payload_kg * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    # Shift to new CoM using parallel axis theorem
    d_old = c6 - coms[6]
    d_new = r - coms[6]
    # I_new_com = I_old_com + m6*(d_old^2 I - d_old d_old^T) + m_p*(d_new^2 I - d_new d_new^T)
    inertias[6] = (
        LINK_INERTIAS[6]
        + m6 * (np.dot(d_old, d_old) * np.eye(3) - np.outer(d_old, d_old))
        + payload_kg * (np.dot(d_new, d_new) * np.eye(3) - np.outer(d_new, d_new))
    )

    return masses, inertias, coms
