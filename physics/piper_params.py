"""Hardcoded Piper 6-DOF parameters for standalone CRBA.

Sourced from binarystars piper_data.py — no DynamicsIR dependency.

The Piper is a lightweight 6-DOF serial arm (~1.3 kg) used in ALOHA
dual-arm teleoperation setups.  Joint axes are all local z, with frame
rotations creating an effective pattern of base-yaw, shoulder-pitch,
elbow-pitch, wrist-roll, wrist-pitch, wrist-roll.

Link 6 combines link6 (7 g) + gripper_base (145 g, fixed joint) into a
single effective end-effector link.  Gripper finger masses (~60 g total)
are excluded as they are on prismatic joints.
"""

from __future__ import annotations

import math

import numpy as np

from physics.spatial import rpy_to_rotation

# ── Raw link parameters from URDF ─────────────────────────────────────────

_LINK_MASSES_RAW = (
    0.215052383265765,    # link1
    0.463914239236335,    # link2 (heaviest)
    0.219942452993132,    # link3
    0.131814339939458,    # link4
    0.134101341225523,    # link5
    0.00699089613564366,  # link6 (very light flange)
)

_GRIPPER_BASE_MASS = 0.145318531013916

_LINK_COMS_RAW = (
    (0.000121504734057468, 0.000104632162460536, -0.00438597309559853),
    (0.198666145229743, -0.010926924140076, 0.00142121714502687),
    (-0.0202737662122021, -0.133914995944595, -0.000458682652737356),
    (-9.66635791618542e-05, 0.000876064475651083, -0.00496880904640868),
    (-4.10554118924211e-05, -0.0566486692356075, -0.0037205791677906),
    (-8.82590762930069e-05, 9.0598378529832e-06, -0.002),
)

_GRIPPER_BASE_COM = (-0.000183807162235591, 8.05033155577911e-05, 0.0321436689908876)

_LINK_INERTIAS_RAW = (
    # link1
    ((0.000109639007860341, 2.50631260865109e-07, -1.89352789149844e-07),
     (2.50631260865109e-07, 9.95612262461418e-05, 1.00634716976093e-08),
     (-1.89352789149844e-07, 1.00634716976093e-08, 0.000116363910317385)),
    # link2
    ((0.000214137415059993, 7.26120579340088e-05, -9.88224861011274e-07),
     (7.26120579340088e-05, 0.00100030277518254, -1.32818212212246e-06),
     (-9.88224861011274e-07, -1.32818212212246e-06, 0.00104417184176783)),
    # link3
    ((0.00018953849076141, -8.05719205057736e-06, 5.10255053956334e-07),
     (-8.05719205057736e-06, 7.1424497082494e-05, 8.89044974368937e-07),
     (5.10255053956334e-07, 8.89044974368937e-07, 0.000201212938725775)),
    # link4
    ((3.96965423235175e-05, -2.32268338444837e-08, -1.14702090783249e-07),
     (-2.32268338444837e-08, 5.13319789853892e-05, 9.92852686264567e-08),
     (-1.14702090783249e-07, 9.92852686264567e-08, 4.14768131680711e-05)),
    # link5
    ((4.10994130543451e-05, -2.06433983793957e-08, 1.29591347668502e-10),
     (-2.06433983793957e-08, 5.27723004189144e-05, 1.9140716904272e-07),
     (1.29591347668502e-10, 1.9140716904272e-07, 4.60418752810541e-05)),
    # link6 (flange only)
    ((5.43015540542155e-07, 0.0, 0.0),
     (0.0, 5.43015540542155e-07, 0.0),
     (0.0, 0.0, 1.06738869138926e-06)),
)

_GRIPPER_BASE_INERTIA = (
    (0.000101740348396288, -1.43961090652723e-07, -8.72352812740139e-08),
    (-1.43961090652723e-07, 4.16518088621566e-05, 3.27712901952435e-08),
    (-8.72352812740139e-08, 3.27712901952435e-08, 0.000118691325723675),
)


# ── Combine link6 + gripper_base into single body ─────────────────────────

def _combine_link6_and_gripper() -> tuple[float, np.ndarray, np.ndarray]:
    """Combine link6 + gripper_base into a single rigid body.

    Returns (mass, com, inertia_about_com) for the combined link6.
    """
    m6 = _LINK_MASSES_RAW[5]
    c6 = np.array(_LINK_COMS_RAW[5])
    I6 = np.array(_LINK_INERTIAS_RAW[5])

    mg = _GRIPPER_BASE_MASS
    cg = np.array(_GRIPPER_BASE_COM)
    Ig = np.array(_GRIPPER_BASE_INERTIA)

    # Combined mass and CoM
    m_total = m6 + mg
    c_combined = (m6 * c6 + mg * cg) / m_total

    # Parallel axis theorem: I_ref = I_com + m * (|d|^2 I - d d^T)
    def _shift_inertia(I_com: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
        return I_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    d6 = c6 - c_combined
    dg = cg - c_combined
    I_combined = _shift_inertia(I6, m6, d6) + _shift_inertia(Ig, mg, dg)

    return m_total, c_combined, I_combined


_m6_combined, _c6_combined, _I6_combined = _combine_link6_and_gripper()


# ── Module-level arrays (link6 = combined link6 + gripper_base) ────────────

LINK_MASSES = np.array(
    list(_LINK_MASSES_RAW[:5]) + [_m6_combined], dtype=float,
)

LINK_COMS = np.zeros((6, 3), dtype=float)
for _i in range(5):
    LINK_COMS[_i] = _LINK_COMS_RAW[_i]
LINK_COMS[5] = _c6_combined

LINK_INERTIAS = np.zeros((6, 3, 3), dtype=float)
for _i in range(5):
    LINK_INERTIAS[_i] = np.array(_LINK_INERTIAS_RAW[_i])
LINK_INERTIAS[5] = _I6_combined

# ── Joint parameters ──────────────────────────────────────────────────────

JOINT_AXES = np.array([
    [0.0, 0.0, 1.0],  # joint1: Z
    [0.0, 0.0, 1.0],  # joint2: Z
    [0.0, 0.0, 1.0],  # joint3: Z
    [0.0, 0.0, 1.0],  # joint4: Z
    [0.0, 0.0, 1.0],  # joint5: Z
    [0.0, 0.0, 1.0],  # joint6: Z
], dtype=float)

JOINT_XYZ = [
    (0.0, 0.0, 0.123),          # joint1
    (0.0, 0.0, 0.0),            # joint2
    (0.28503, 0.0, 0.0),        # joint3
    (-0.021984, -0.25075, 0.0), # joint4
    (0.0, 0.0, 0.0),            # joint5
    (8.8259e-05, -0.091, 0.0),  # joint6
]

JOINT_RPY = [
    (0.0, 0.0, 0.0),            # joint1
    (1.5708, -0.10095, -3.1416), # joint2
    (0.0, 0.0, -1.759),         # joint3
    (1.5708, 0.0, 0.0),         # joint4
    (-1.5708, 0.0, 0.0),        # joint5
    (1.5708, 0.0, 0.0),         # joint6
]

PARENT_INDICES = (-1, 0, 1, 2, 3, 4)

N_JOINTS = 6

# EE offset from last joint frame (gripper tip approx)
EE_OFFSET = np.array([0.0, 0.0, 0.07], dtype=float)


# ── Build transforms ──────────────────────────────────────────────────────

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


# ── Convenience functions ─────────────────────────────────────────────────

def compute_piper_mass_matrix(q: np.ndarray) -> np.ndarray:
    """Convenience: compute M(q) for the standard Piper 6-DOF."""
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


def compute_piper_coupling(q: np.ndarray) -> np.ndarray:
    """Convenience: compute J_ij(q) coupling matrix for Piper."""
    from physics.coupling import normalized_coupling_matrix
    M = compute_piper_mass_matrix(q)
    return normalized_coupling_matrix(M)


def modify_payload(payload_kg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return modified (masses, inertias, coms) with payload at EE.

    Payload m_p attached at EE offset from link6 frame.
    """
    masses = LINK_MASSES.copy()
    inertias = LINK_INERTIAS.copy()
    coms = LINK_COMS.copy()

    m5 = masses[5]
    c5 = coms[5].copy()
    r = EE_OFFSET

    # Update mass
    masses[5] = m5 + payload_kg

    # Update CoM
    coms[5] = (m5 * c5 + payload_kg * r) / (m5 + payload_kg)

    # Update inertia via parallel axis theorem about new CoM
    d_old = c5 - coms[5]
    d_new = r - coms[5]
    inertias[5] = (
        LINK_INERTIAS[5]
        + m5 * (np.dot(d_old, d_old) * np.eye(3) - np.outer(d_old, d_old))
        + payload_kg * (np.dot(d_new, d_new) * np.eye(3) - np.outer(d_new, d_new))
    )

    return masses, inertias, coms
