"""Entanglement propagation time analysis.

Computes arrival time of entanglement from a source qubit to all others.
Key insight (Exp B'): propagation time correlates with |J_ij|, NOT kinematic distance.

Uses single-excitation sector for efficiency (n-dimensional, exact).
"""

from __future__ import annotations

import numpy as np

from quantum_prior.entanglement_graph import (
    SingleExcitationEvolver,
    concurrence_from_amplitudes,
    local_field_terms,
    normalized_coupling_matrix,
    single_excitation_hamiltonian,
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

    # Single-excitation sector: n×n
    H_eff = single_excitation_hamiltonian(J, h)
    evolver = SingleExcitationEvolver(H_eff)

    # Initial state: excite source qubit
    c0 = np.zeros(n, dtype=complex)
    c0[source_qubit] = 1.0

    times = np.linspace(0, t_max, n_time_steps)
    states = evolver.evolve_series(c0, times)

    arrival_times = {}
    for t_idx, c_t in enumerate(states):
        for j in range(n):
            if j == source_qubit or j in arrival_times:
                continue
            c = concurrence_from_amplitudes(c_t, source_qubit, j)
            if c > threshold:
                arrival_times[j] = float(times[t_idx])

    return arrival_times
