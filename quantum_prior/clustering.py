"""Entanglement spectral clustering.

Ported from experiments/structure_prior_analysis.py:131-181.

Groups joints by entanglement strength using normalized Laplacian eigenvectors.
Key advantage over threshold-based methods: clean spectral gap, sharp phase transitions.
"""

from __future__ import annotations

import numpy as np


def spectral_clustering(W: np.ndarray, n_clusters: int) -> list[int]:
    """Spectral clustering on weighted adjacency matrix W.

    Uses the normalized Laplacian and k-means on the first k eigenvectors.

    Parameters
    ----------
    W : (n, n) non-negative symmetric weight matrix
    n_clusters : number of clusters

    Returns
    -------
    labels : list of cluster assignments (length n)
    """
    n = W.shape[0]
    if n_clusters >= n:
        return list(range(n))

    D = np.diag(np.sum(np.abs(W), axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-15)))
    L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    # Use first k eigenvectors (smallest eigenvalues)
    V = eigenvectors[:, :n_clusters]
    # Normalize rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    V = V / norms

    # Simple k-means (few iterations suffice for small n)
    rng = np.random.RandomState(42)
    centers = V[rng.choice(n, n_clusters, replace=False)]
    for _ in range(50):
        dists = np.array([np.linalg.norm(V - c, axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        new_centers = np.array(
            [
                V[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(n_clusters)
            ]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels.tolist()


def optimal_n_clusters(W: np.ndarray) -> int:
    """Find optimal number of clusters from spectral gap of Laplacian.

    Returns
    -------
    k : optimal cluster count (>= 2)
    """
    n = W.shape[0]
    D = np.diag(np.sum(np.abs(W), axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-15)))
    L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))

    # Find largest gap between consecutive eigenvalues (skip first which is ~0)
    gaps = np.diff(eigenvalues[1:])
    if len(gaps) == 0:
        return 1
    best_k = int(np.argmax(gaps)) + 2
    return max(2, min(best_k, n - 1))


def decompose_joints(C_matrix: np.ndarray) -> list[list[int]]:
    """Decompose joints into groups using entanglement spectral clustering.

    Parameters
    ----------
    C_matrix : (n, n) concurrence matrix from compute_entanglement_graph

    Returns
    -------
    groups : list of lists, each containing joint indices in one cluster
    """
    k = optimal_n_clusters(C_matrix)
    labels = spectral_clustering(C_matrix, k)
    groups = []
    for cluster_id in range(k):
        group = [i for i, lab in enumerate(labels) if lab == cluster_id]
        if group:
            groups.append(sorted(group))
    return groups
