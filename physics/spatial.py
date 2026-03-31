"""6D spatial algebra utilities (Featherstone [omega, v] convention).

Standalone port from binarystars core — no external dependencies beyond numpy.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def skew(v: FloatArray) -> FloatArray:
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=float,
    )


def spatial_inertia(mass: float, inertia: np.ndarray, com: np.ndarray) -> FloatArray:
    """Construct 6x6 spatial inertia matrix.

    Featherstone [omega, v] convention:
        I_s = [[I_rot + m*cx*cx^T,  m*cx],
               [m*cx^T,              m*I_3]]
    """
    cx = skew(com)
    I3 = np.eye(3, dtype=float)
    top_left = inertia + mass * (cx @ cx.T)
    top_right = mass * cx
    bottom_left = mass * cx.T
    bottom_right = mass * I3
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def spatial_cross_star(v: np.ndarray) -> FloatArray:
    """Spatial force cross-product operator v x* (6x6)."""
    omega = v[:3]
    v_lin = v[3:]
    sw = skew(omega)
    sv = skew(v_lin)
    Z = np.zeros((3, 3), dtype=float)
    return np.block([[sw, sv], [Z, sw]])


def spatial_transform_inverse(
    rotation: np.ndarray, translation: np.ndarray
) -> FloatArray:
    """6x6 spatial transform from child frame to parent frame (inverse adjoint).

    Featherstone:  X = [[R, 0], [-R*px, R]]
    where px = skew(translation), R = child-to-parent rotation.
    """
    R = np.asarray(rotation, dtype=float)
    p = np.asarray(translation, dtype=float)
    px = skew(p)
    Z = np.zeros((3, 3), dtype=float)
    return np.block([[R, Z], [-R @ px, R]])


def rotation_about_axis(axis: np.ndarray, angle: float) -> FloatArray:
    """Rodrigues rotation: 3x3 matrix for rotation about unit axis by angle (rad)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    K = skew(axis)
    return np.eye(3, dtype=float) + s * K + (1 - c) * (K @ K)


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> FloatArray:
    """RPY to rotation matrix (ROS convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)
