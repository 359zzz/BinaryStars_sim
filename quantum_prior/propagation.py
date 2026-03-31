"""Entanglement propagation time analysis.

Computes arrival time of entanglement from a source qubit to all others.
Key insight (Exp B'): propagation time correlates with |J_ij|, NOT kinematic distance.
"""

from __future__ import annotations

import numpy as np

from quantum_prior.entanglement_graph import (
    EigenEvolver,
    heisenberg_hamiltonian,
    local_field_terms,
    normalized_coupling_matrix,
    pairwise_concurrence,
)


def compute_propagation_times(
    M: np.ndarray,
    source_qubit: int = 0,
    t_max: float = 3.0,
    n_time_steps: int = 200,
    threshold: float = 0.05,
) -> dict[int, float]:
    """Compute entanglement propagation times from source qubit.

    Parameters
    ----------
    M : (n, n) mass matrix
    source_qubit : qubit index from which excitation propagates
    t_max : maximum evolution time
    n_time_steps : temporal resolution
    threshold : concurrence threshold for "arrival"

    Returns
    -------
    arrival_times : dict mapping joint index -> first time C > threshold.
                    Missing keys = never reached threshold.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]

    J = normalized_coupling_matrix(M)
    h = local_field_terms(M)
    H = heisenberg_hamiltonian(J, h)

    # Initial state: excite source qubit
    dim = 2**n
    psi0 = np.zeros(dim, dtype=complex)
    # |0...1_source...0> = set bit at source_qubit position
    psi0[1 << (n - 1 - source_qubit)] = 1.0

    evolver = EigenEvolver(H)
    times = np.linspace(0, t_max, n_time_steps)
    states = evolver.evolve_series(psi0, times)

    arrival_times = {}
    for t_idx, psi_t in enumerate(states):
        for j in range(n):
            if j == source_qubit or j in arrival_times:
                continue
            c = pairwise_concurrence(psi_t, source_qubit, j, n)
            if c > threshold:
                arrival_times[j] = float(times[t_idx])

    return arrival_times
