"""Quantum structure prior module.

Self-contained numpy/scipy implementation of:
- Entanglement graph computation (M -> C_ij via Heisenberg Hamiltonian)
- LRU-cached entanglement computer for RL integration
- Entanglement spectral clustering
- Entanglement propagation time analysis
"""

from quantum_prior.entanglement_graph import (
    compute_classical_features,
    compute_entanglement_features,
    compute_entanglement_graph,
)
from quantum_prior.cached_computer import CachedEntanglementComputer
from quantum_prior.clustering import (
    decompose_joints,
    optimal_n_clusters,
    spectral_clustering,
)

__all__ = [
    "compute_entanglement_graph",
    "compute_entanglement_features",
    "compute_classical_features",
    "CachedEntanglementComputer",
    "spectral_clustering",
    "optimal_n_clusters",
    "decompose_joints",
]
