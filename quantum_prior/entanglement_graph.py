"""Core quantum entanglement graph computation.

Pipeline: M(q) -> (J, h) -> H_M -> EigenEvolver -> evolve |0...01> ->
          pairwise concurrences -> max over t -> C_matrix.

Ported from:
- quantum/dynamics_hamiltonian.py (Pauli matrices, Heisenberg Hamiltonian)
- quantum/time_evolution.py (EigenEvolver)
- quantum/entanglement_measures.py (concurrence, pairwise concurrence)
- dynamics/coupling.py (normalized coupling, local fields)
"""

from __future__ import annotations

import numpy as np

# ── Pauli matrices ────────────────────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_YY = np.kron(_Y, _Y)


# ── Coupling from mass matrix ────────────────────────────────────────────────

def normalized_coupling_matrix(M: np.ndarray) -> np.ndarray:
    """J_ij = M_ij / sqrt(M_ii * M_jj), n x n symmetric, J_ii = 1."""
    M = np.asarray(M, dtype=float)
    diag = np.diag(M)
    if np.any(diag <= 0):
        raise ValueError("mass matrix diagonal must be strictly positive")
    scale = np.sqrt(np.outer(diag, diag))
    return M / scale


def local_field_terms(M: np.ndarray) -> np.ndarray:
    """h_i = M_ii / (2 * max(diag(M))), normalized to [0, 0.5]."""
    M = np.asarray(M, dtype=float)
    diag = np.diag(M)
    if np.any(diag <= 0):
        raise ValueError("mass matrix diagonal must be strictly positive")
    return diag / (2.0 * np.max(diag))


# ── Hamiltonian construction ──────────────────────────────────────────────────

def _kron_identity(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Place single-qubit operator at position `qubit` in n-qubit space."""
    parts = [_I2] * n_qubits
    parts[qubit] = op
    result = parts[0]
    for p in parts[1:]:
        result = np.kron(result, p)
    return result


def _two_qubit_pauli_term(
    pauli: np.ndarray, i: int, j: int, n_qubits: int
) -> np.ndarray:
    """Tensor product: I...pauli_i...pauli_j...I."""
    parts = [_I2] * n_qubits
    parts[i] = pauli
    parts[j] = pauli
    result = parts[0]
    for p in parts[1:]:
        result = np.kron(result, p)
    return result


def heisenberg_hamiltonian(
    J: np.ndarray, h: np.ndarray, *, max_qubits: int = 12
) -> np.ndarray:
    """Build Heisenberg Hamiltonian from coupling matrix J and local fields h.

    H = sum_{i<j} J_ij (X_i X_j + Y_i Y_j + Z_i Z_j) + sum_i h_i Z_i
    """
    n = J.shape[0]
    if n > max_qubits:
        raise ValueError(f"n={n} exceeds max_qubits={max_qubits}")
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)

    # Exchange terms
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) < 1e-15:
                continue
            for pauli in (_X, _Y, _Z):
                H += J[i, j] * _two_qubit_pauli_term(pauli, i, j, n)

    # Local field terms
    for i in range(n):
        if abs(h[i]) < 1e-15:
            continue
        H += h[i] * _kron_identity(_Z, i, n)

    return H


# ── Time evolution via eigendecomposition ─────────────────────────────────────

class EigenEvolver:
    """Efficient time evolution: diagonalize once, evolve O(dim) per time step."""

    def __init__(self, H: np.ndarray) -> None:
        H = np.asarray(H, dtype=complex)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(H)

    def evolve(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """Return |psi(t)> = V diag(e^{-iEt}) V^dag |psi0>."""
        coeffs = self.eigenvectors.conj().T @ psi0
        phases = np.exp(-1j * self.eigenvalues * t)
        return self.eigenvectors @ (phases * coeffs)

    def evolve_series(
        self, psi0: np.ndarray, times: np.ndarray
    ) -> list[np.ndarray]:
        """Evolve at multiple times efficiently."""
        psi0 = np.asarray(psi0, dtype=complex)
        coeffs = self.eigenvectors.conj().T @ psi0
        return [
            self.eigenvectors @ (np.exp(-1j * self.eigenvalues * t) * coeffs)
            for t in times
        ]


# ── Entanglement measures ─────────────────────────────────────────────────────

def concurrence_mixed(rho: np.ndarray) -> float:
    """Wootters concurrence for a 2-qubit mixed state (4x4 density matrix).

    C = max(0, sqrt(l1) - sqrt(l2) - sqrt(l3) - sqrt(l4))
    """
    rho = np.asarray(rho, dtype=complex)
    R = rho @ _YY @ rho.conj() @ _YY
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(R).real))[::-1]
    sqrt_eigs = np.sqrt(np.maximum(eigenvalues, 0.0))
    return float(max(0.0, sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3]))


def pairwise_concurrence(
    psi: np.ndarray, i: int, j: int, n_qubits: int
) -> float:
    """Concurrence between qubits i and j from pure state psi.

    Uses tensor contraction: reshape to (4, 2^(n-2)), compute rho_ij = M @ M^dag.
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    if i > j:
        i, j = j, i
    tensor = psi.reshape([2] * n_qubits)
    axes = list(range(n_qubits))
    axes.remove(i)
    axes.remove(j)
    new_axes = [i, j] + axes
    tensor = tensor.transpose(new_axes)
    M_mat = tensor.reshape(4, -1)
    rho_ij = M_mat @ M_mat.conj().T
    return concurrence_mixed(rho_ij)


def all_pairwise_concurrences(
    psi: np.ndarray, n_qubits: int
) -> dict[tuple[int, int], float]:
    """Compute concurrence for all pairs (i < j)."""
    result = {}
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            result[(i, j)] = pairwise_concurrence(psi, i, j, n_qubits)
    return result


# ── Main API ──────────────────────────────────────────────────────────────────

def compute_entanglement_graph(
    M: np.ndarray,
    t_max: float = 3.0,
    n_time_steps: int = 50,
) -> np.ndarray:
    """Compute entanglement graph from mass matrix M(q).

    Returns (n, n) matrix where C_ij = max_t concurrence(psi(t), i, j).
    Initial state: |0...01> (last qubit excited).

    Parameters
    ----------
    M : (n, n) mass matrix at configuration q
    t_max : maximum evolution time
    n_time_steps : number of time points to sample

    Returns
    -------
    C_matrix : (n, n) symmetric, C_ii = 0, C_ij in [0, 1]
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]

    # Coupling and local fields from mass matrix
    J = normalized_coupling_matrix(M)
    h = local_field_terms(M)

    # Build Hamiltonian
    H = heisenberg_hamiltonian(J, h)

    # Initial state |0...01> (last qubit excited)
    dim = 2**n
    psi0 = np.zeros(dim, dtype=complex)
    psi0[1] = 1.0  # |0...01> = basis index 1 (binary: ...001)

    # Time evolution
    evolver = EigenEvolver(H)
    times = np.linspace(0, t_max, n_time_steps)
    states = evolver.evolve_series(psi0, times)

    # Max concurrence over time for each pair
    C_matrix = np.zeros((n, n))
    for psi_t in states:
        concurrences = all_pairwise_concurrences(psi_t, n)
        for (i, j), c in concurrences.items():
            if c > C_matrix[i, j]:
                C_matrix[i, j] = c
                C_matrix[j, i] = c

    return C_matrix


def compute_entanglement_features(M: np.ndarray, **kwargs) -> np.ndarray:
    """C_matrix upper-triangle flattened -> (n*(n-1)/2,) vector.

    For n=7: (21,) vector.
    """
    C = compute_entanglement_graph(M, **kwargs)
    n = C.shape[0]
    features = []
    for i in range(n):
        for j in range(i + 1, n):
            features.append(C[i, j])
    return np.array(features, dtype=np.float32)


def compute_classical_features(M: np.ndarray) -> np.ndarray:
    """|J_ij| upper-triangle flattened -> (n*(n-1)/2,) vector.

    For n=7: (21,) vector. Classical pairwise coupling (no quantum dynamics).
    """
    J = normalized_coupling_matrix(M)
    n = J.shape[0]
    features = []
    for i in range(n):
        for j in range(i + 1, n):
            features.append(abs(J[i, j]))
    return np.array(features, dtype=np.float32)
