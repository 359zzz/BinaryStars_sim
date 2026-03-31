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


# ── Single-excitation sector (exact for |0...01⟩ initial state) ──────────────

def single_excitation_hamiltonian(
    J: np.ndarray, h: np.ndarray,
) -> np.ndarray:
    """Build n×n effective Hamiltonian in the single-excitation sector.

    The Heisenberg Hamiltonian preserves total z-magnetization.
    Starting from |0...01⟩ (HW=1), dynamics stays in the n-dimensional
    subspace span{|e_0⟩, ..., |e_{n-1}⟩} where |e_i⟩ has qubit i excited.

    This reduces 2^n to n-dimensional eigenvalue problem — exact, not
    an approximation.

    Matrix elements:
        H_eff[i,j] = 2·J_ij           (i ≠ j, from XX+YY exchange)
        H_eff[i,i] = -2·Σ_{j≠i} J_ij - 2·h_i + const
    """
    n = J.shape[0]
    H_eff = np.zeros((n, n), dtype=float)

    # Off-diagonal: exchange coupling
    for i in range(n):
        for j in range(i + 1, n):
            H_eff[i, j] = 2.0 * J[i, j]
            H_eff[j, i] = 2.0 * J[i, j]

    # Diagonal: ZZ terms + local fields
    # Constant shift: Σ_{a<b} J_ab + Σ_a h_a (same for all i)
    const = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            const += J[a, b]
        const += h[a]

    for i in range(n):
        off_sum = sum(J[i, j] for j in range(n) if j != i)
        H_eff[i, i] = -2.0 * off_sum - 2.0 * h[i] + const

    return H_eff


class SingleExcitationEvolver:
    """Efficient evolution in the n-dimensional single-excitation sector."""

    def __init__(self, H_eff: np.ndarray) -> None:
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(H_eff)

    def evolve(self, c0: np.ndarray, t: float) -> np.ndarray:
        """Return c(t) in the n-dim sector."""
        coeffs = self.eigenvectors.conj().T @ c0
        phases = np.exp(-1j * self.eigenvalues * t)
        return self.eigenvectors @ (phases * coeffs)

    def evolve_series(
        self, c0: np.ndarray, times: np.ndarray,
    ) -> list[np.ndarray]:
        coeffs = self.eigenvectors.conj().T @ c0
        return [
            self.eigenvectors @ (np.exp(-1j * self.eigenvalues * t) * coeffs)
            for t in times
        ]


def concurrence_from_amplitudes(c: np.ndarray, i: int, j: int) -> float:
    """Exact concurrence for single-excitation state: C_ij = 2|c_i||c_j|."""
    return 2.0 * abs(c[i]) * abs(c[j])


def entropy_from_amplitudes(c: np.ndarray, n_L: int) -> float:
    """Bipartite entropy for L|R split of single-excitation state.

    In HW=1 sector, the entanglement spectrum is binary: (p_L, p_R).
    """
    p_L = float(np.sum(np.abs(c[:n_L])**2))
    p_R = float(np.sum(np.abs(c[n_L:])**2))
    if p_L < 1e-15 or p_R < 1e-15:
        return 0.0
    return -p_L * np.log2(p_L) - p_R * np.log2(p_R)


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

    Uses the single-excitation sector (n-dimensional, exact) for efficiency.
    This reduces the problem from 2^n to n dimensions.

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

    J = normalized_coupling_matrix(M)
    h = local_field_terms(M)

    # Single-excitation sector: n×n instead of 2^n × 2^n
    H_eff = single_excitation_hamiltonian(J, h)
    evolver = SingleExcitationEvolver(H_eff)

    # Initial state: last qubit excited
    c0 = np.zeros(n, dtype=complex)
    c0[n - 1] = 1.0

    times = np.linspace(0, t_max, n_time_steps)
    states = evolver.evolve_series(c0, times)

    # Max concurrence over time: C_ij = 2|c_i||c_j|
    C_matrix = np.zeros((n, n))
    for c_t in states:
        for i in range(n):
            for j in range(i + 1, n):
                c = concurrence_from_amplitudes(c_t, i, j)
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


# ── Bipartite entanglement spectrum & entropy ────────────────────────────────

def bipartite_reduced_density_matrix(
    psi: np.ndarray, n_L: int, n_R: int,
) -> np.ndarray:
    """Compute reduced density matrix ρ_L = Tr_R(|ψ⟩⟨ψ|).

    Parameters
    ----------
    psi : state vector of length 2^(n_L + n_R)
    n_L : number of qubits in subsystem L (left arm)
    n_R : number of qubits in subsystem R (right arm)

    Returns
    -------
    rho_L : (2^n_L, 2^n_L) complex Hermitian matrix
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    dim_L = 2**n_L
    dim_R = 2**n_R
    Psi = psi.reshape(dim_L, dim_R)
    return Psi @ Psi.conj().T


def entanglement_spectrum(
    psi: np.ndarray, n_L: int, n_R: int,
) -> np.ndarray:
    """Eigenvalues of ρ_L sorted in decreasing order.

    These are the Schmidt coefficients squared. For a product state,
    only one eigenvalue is nonzero (= 1). For a maximally entangled
    state of min(n_L, n_R) qubits, eigenvalues are uniform.

    Returns
    -------
    spectrum : (2^n_L,) array, sorted descending, sums to 1
    """
    rho_L = bipartite_reduced_density_matrix(psi, n_L, n_R)
    eigenvalues = np.linalg.eigvalsh(rho_L).real
    # Clamp numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)
    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues


def bipartite_entanglement_entropy(
    psi: np.ndarray, n_L: int, n_R: int,
) -> float:
    """von Neumann entropy S(ρ_L) = -Σ λ_k log₂ λ_k.

    Returns 0.0 for product states, log₂(min(2^n_L, 2^n_R)) for
    maximally entangled states.
    """
    spec = entanglement_spectrum(psi, n_L, n_R)
    # Filter out zeros to avoid log(0)
    nonzero = spec[spec > 1e-15]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def spectral_distance(
    spec1: np.ndarray, spec2: np.ndarray,
    S1: float | None = None, S2: float | None = None,
) -> float:
    """Entanglement spectral distance D(R₁, R₂).

    D = |S₁ - S₂| + ||λ̃₁ - λ̃₂||₂

    where λ̃ᵢ is zero-padded to equal length.

    Parameters
    ----------
    spec1, spec2 : entanglement spectra (sorted descending)
    S1, S2 : precomputed entropies (optional, computed from spectra if None)
    """
    spec1 = np.asarray(spec1)
    spec2 = np.asarray(spec2)

    # Zero-pad to equal length
    max_len = max(len(spec1), len(spec2))
    s1 = np.zeros(max_len)
    s2 = np.zeros(max_len)
    s1[:len(spec1)] = spec1
    s2[:len(spec2)] = spec2

    # Entropy difference
    if S1 is None:
        nz = spec1[spec1 > 1e-15]
        S1 = float(-np.sum(nz * np.log2(nz))) if len(nz) > 0 else 0.0
    if S2 is None:
        nz = spec2[spec2 > 1e-15]
        S2 = float(-np.sum(nz * np.log2(nz))) if len(nz) > 0 else 0.0

    return abs(S1 - S2) + float(np.linalg.norm(s1 - s2))


def compute_entanglement_spectrum_from_mass_matrix(
    M: np.ndarray,
    n_L: int,
    n_R: int,
    t_star: float | None = None,
) -> dict:
    """Full pipeline: M(q) → Hamiltonian → evolve → entanglement spectrum.

    Uses the single-excitation sector for efficiency (n×n instead of
    2^n × 2^n).  Exact for the standard |0...01⟩ initial state.

    Parameters
    ----------
    M : (n, n) mass matrix where n = n_L + n_R
    n_L : number of joints in left arm
    n_R : number of joints in right arm
    t_star : evolution time (default: π/(4·max|J_ij|))

    Returns
    -------
    dict with keys: 'spectrum', 'entropy', 'J_matrix', 'h_vector',
                    'psi', 't_star', 'p_L', 'p_R'
    """
    n = M.shape[0]
    assert n == n_L + n_R, f"n={n} != n_L+n_R={n_L+n_R}"

    J = normalized_coupling_matrix(M)
    h = local_field_terms(M)

    # Characteristic time
    J_off = np.abs(J.copy())
    np.fill_diagonal(J_off, 0.0)
    max_J = np.max(J_off)
    if t_star is None:
        if max_J < 1e-15:
            t_star = 1.0
        else:
            t_star = np.pi / (4.0 * max_J)

    # Single-excitation sector: n×n
    H_eff = single_excitation_hamiltonian(J, h)
    evolver = SingleExcitationEvolver(H_eff)

    c0 = np.zeros(n, dtype=complex)
    c0[n - 1] = 1.0
    c_t = evolver.evolve(c0, t_star)

    # Bipartite probabilities
    p_L = float(np.sum(np.abs(c_t[:n_L])**2))
    p_R = float(np.sum(np.abs(c_t[n_L:])**2))

    # Entropy
    S = entropy_from_amplitudes(c_t, n_L)

    # Spectrum (binary in single-excitation sector)
    dim_L = 2**n_L
    spec = np.zeros(dim_L)
    spec[0] = max(p_L, p_R)
    spec[1] = min(p_L, p_R)

    # Reconstruct full 2^n state for compatibility
    dim = 2**n
    psi_full = np.zeros(dim, dtype=complex)
    for i in range(n):
        # |e_i⟩: qubit i excited → bit (n-1-i) set → index 2^{n-1-i}
        idx = 1 << (n - 1 - i)
        psi_full[idx] = c_t[i]

    return {
        'spectrum': spec,
        'entropy': S,
        'J_matrix': J,
        'h_vector': h,
        'psi': psi_full,
        't_star': t_star,
        'p_L': p_L,
        'p_R': p_R,
    }


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
