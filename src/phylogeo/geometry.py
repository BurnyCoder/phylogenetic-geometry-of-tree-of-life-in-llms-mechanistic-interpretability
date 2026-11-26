"""Representation geometry: species vectors, cosine distances, kNN-graph geodesics.

Global context: turns raw activations [n_names, n_templates, n_layers + 1, d] into
per-layer species vectors (preregistered normalization: subtract the mean over
species within each template, then average templates), and provides the
distances compared with the tree. The kNN geodesic follows Modell et al. 2025
(https://arxiv.org/abs/2505.18235) and Goodfire's Evo 2 analysis: build a kNN
graph with angular edge weights at the smallest k that makes it connected and
take shortest-path lengths (Dijkstra, scipy.sparse.csgraph).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def species_vectors(X: np.ndarray, center: bool = True) -> np.ndarray:
    """[n, T, L, d] activations -> [L, n, d] species vectors.

    center=True: per template, subtract the mean over species (removes the shared
    template/anisotropy direction), then average over templates. center=False:
    plain template average ("raw" sensitivity variant).
    """
    Y = X - X.mean(axis=0, keepdims=True) if center else X
    return Y.mean(axis=1).transpose(1, 0, 2)


def zscore_dims(V: np.ndarray) -> np.ndarray:
    """Standardize each dimension over species (sensitivity variant for rogue dims)."""
    mu = V.mean(-2, keepdims=True)
    sd = V.std(-2, keepdims=True) + 1e-8
    return (V - mu) / sd


def cosine_distance(V: np.ndarray) -> np.ndarray:
    """Cosine distance 1 - cos for [n, d] or [L, n, d]; exact zeros on the diagonal."""
    U = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-12)
    D = 1.0 - U @ np.swapaxes(U, -1, -2)
    D = np.clip(D, 0.0, 2.0)
    idx = np.arange(D.shape[-1])
    D[..., idx, idx] = 0.0
    return D


def angular_distance(V: np.ndarray) -> np.ndarray:
    """arccos(cos) in radians (Goodfire's edge weight); monotone in cosine distance."""
    return np.arccos(np.clip(1.0 - cosine_distance(V), -1.0, 1.0))


def knn_graph(D: np.ndarray, k: int) -> csr_matrix:
    """Symmetric kNN graph: edge i-j if j is among i's k nearest or vice versa, weight D[i, j]."""
    n = D.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be between one and the number of other species")
    ranked = np.array(D, dtype=float, copy=True)
    np.fill_diagonal(ranked, np.inf)
    neighbours = np.argsort(ranked, axis=1, kind="stable")[:, :k]
    keep = np.zeros((n, n), dtype=bool)
    keep[np.arange(n)[:, None], neighbours] = True
    a, b = np.where(keep | keep.T)
    # An explicitly stored zero is a valid edge between identical representations.
    # https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
    return csr_matrix((D[a, b], (a, b)), shape=D.shape)


def smallest_connected_k(D: np.ndarray, k_max: int | None = None) -> int:
    """Smallest k for which the symmetric kNN graph is connected (Modell et al.; Goodfire K=27)."""
    n = D.shape[0]
    for k in range(1, (k_max or n - 1) + 1):
        if connected_components(knn_graph(D, k), directed=False)[0] == 1:
            return k
    return n - 1


def knn_geodesic(D: np.ndarray, k: int | None = None) -> tuple[np.ndarray, int]:
    """Shortest-path distances on the kNN graph (k = smallest connected k by default)."""
    k = k or smallest_connected_k(D)
    G = shortest_path(knn_graph(D, k), method="D", directed=False)
    return G, k


def split_half_reliability(X: np.ndarray, rng: np.random.Generator, n_splits: int = 20) -> np.ndarray:
    """Split-half template reliability per layer: Spearman between distance matrices from two random
    template halves, Spearman-Brown corrected to the full template count. Returns [L]."""
    from phylogeo.stats.mantel import mantel_r

    T = X.shape[1]
    out = []
    for _ in range(n_splits):
        p = rng.permutation(T)
        A = cosine_distance(species_vectors(X[:, p[: T // 2]]))
        B = cosine_distance(species_vectors(X[:, p[T // 2 :]]))
        r = np.array([mantel_r(A[l], B[l]) for l in range(A.shape[0])])
        out.append(2 * r / (1 + r))
    return np.mean(out, axis=0)
