"""Standalone physics module — pure numpy, no quantum SDK dependencies."""

from physics.spatial import (
    skew,
    spatial_inertia,
    spatial_cross_star,
    spatial_transform_inverse,
    rotation_about_axis,
)
from physics.crba import compute_mass_matrix
from physics.coupling import normalized_coupling_matrix, local_field_terms

__all__ = [
    "skew",
    "spatial_inertia",
    "spatial_cross_star",
    "spatial_transform_inverse",
    "rotation_about_axis",
    "compute_mass_matrix",
    "normalized_coupling_matrix",
    "local_field_terms",
]
