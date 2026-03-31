"""LRU-cached entanglement computer for RL integration.

Discretizes q to `resolution` radians, caches entanglement graphs per
discretized configuration. For smooth trajectories, cache hit rate >95%.

Performance (n=7): ~0.7ms per unique q.
PPO 16384 steps * 5% miss = ~573ms/update (negligible overhead).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np

from quantum_prior.entanglement_graph import (
    compute_entanglement_graph,
    normalized_coupling_matrix,
)


class CachedEntanglementComputer:
    """Cached entanglement computation with q-space discretization.

    Parameters
    ----------
    mass_matrix_fn : callable q -> M(q), returns (n, n) mass matrix
    resolution : discretization resolution in radians (default 0.01)
    cache_size : LRU cache size (default 65536)
    t_max : max evolution time for entanglement graph
    n_time_steps : number of time points
    """

    def __init__(
        self,
        mass_matrix_fn: Callable[[np.ndarray], np.ndarray],
        resolution: float = 0.01,
        cache_size: int = 65536,
        t_max: float = 3.0,
        n_time_steps: int = 50,
    ):
        self._mass_matrix_fn = mass_matrix_fn
        self._resolution = resolution
        self._t_max = t_max
        self._n_time_steps = n_time_steps

        self._hits = 0
        self._misses = 0

        # Create cached versions
        self._cached_entanglement = lru_cache(maxsize=cache_size)(
            self._compute_entanglement_graph_impl
        )
        self._cached_classical = lru_cache(maxsize=cache_size)(
            self._compute_classical_coupling_impl
        )

    def _discretize(self, q: np.ndarray) -> tuple:
        """Round q to resolution and convert to hashable tuple."""
        q_discrete = np.round(np.asarray(q) / self._resolution) * self._resolution
        return tuple(q_discrete.tolist())

    def _compute_entanglement_graph_impl(self, q_key: tuple) -> tuple:
        """Internal: compute and return as tuple (for lru_cache hashability)."""
        q = np.array(q_key)
        M = self._mass_matrix_fn(q)
        C = compute_entanglement_graph(M, self._t_max, self._n_time_steps)
        return tuple(C.ravel().tolist())

    def _compute_classical_coupling_impl(self, q_key: tuple) -> tuple:
        """Internal: compute classical coupling matrix."""
        q = np.array(q_key)
        M = self._mass_matrix_fn(q)
        J = normalized_coupling_matrix(M)
        return tuple(J.ravel().tolist())

    def get_entanglement_graph(self, q: np.ndarray) -> np.ndarray:
        """Return (n, n) concurrence matrix C_ij for configuration q."""
        q_key = self._discretize(q)
        info = self._cached_entanglement.cache_info()
        result_tuple = self._cached_entanglement(q_key)
        new_info = self._cached_entanglement.cache_info()
        if new_info.hits > info.hits:
            self._hits += 1
        else:
            self._misses += 1
        n = len(q_key)
        return np.array(result_tuple, dtype=np.float64).reshape(n, n)

    def get_entanglement_features(self, q: np.ndarray) -> np.ndarray:
        """Return upper-triangle C_ij as flat vector. (n=7 -> 21-d)."""
        C = self.get_entanglement_graph(q)
        n = C.shape[0]
        features = []
        for i in range(n):
            for j in range(i + 1, n):
                features.append(C[i, j])
        return np.array(features, dtype=np.float32)

    def get_classical_features(self, q: np.ndarray) -> np.ndarray:
        """Return upper-triangle |J_ij| as flat vector. (n=7 -> 21-d)."""
        J = self.get_classical_coupling(q)
        n = J.shape[0]
        features = []
        for i in range(n):
            for j in range(i + 1, n):
                features.append(abs(J[i, j]))
        return np.array(features, dtype=np.float32)

    def get_classical_coupling(self, q: np.ndarray) -> np.ndarray:
        """Return (n, n) signed coupling matrix J_ij for configuration q."""
        q_key = self._discretize(q)
        result_tuple = self._cached_classical(q_key)
        n = len(q_key)
        return np.array(result_tuple, dtype=np.float64).reshape(n, n)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def cache_info(self) -> dict:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "entanglement_cache": str(self._cached_entanglement.cache_info()),
            "classical_cache": str(self._cached_classical.cache_info()),
        }
