"""Normalized coupling coefficients from mass matrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def normalized_coupling_matrix(M: np.ndarray) -> FloatArray:
    """J_ij = M_ij / sqrt(M_ii * M_jj), symmetric, J_ii = 1."""
    M = np.asarray(M, dtype=float)
    diag = np.diag(M)
    scale = np.sqrt(np.outer(diag, diag))
    return M / scale


def local_field_terms(M: np.ndarray) -> FloatArray:
    """h_i = M_ii / (2 * max(diag(M))), normalized to [0, 0.5]."""
    M = np.asarray(M, dtype=float)
    diag = np.diag(M)
    max_diag = np.max(diag)
    return diag / (2.0 * max_diag)
