"""Mantel tests between distance matrices, with max-T over layers and partial Mantel.

Global context: secondary hypothesis S1 (preregistration) is a Goodfire-comparable
global correlation between representation distance and tree distance. Pairwise
distances are not independent, so significance comes from permuting the rows
and columns of one matrix together (Mantel 1967). The ordinary statistic matches
``skbio.stats.distance.mantel`` (tested); this module adds the max-T variant
across layers (one permutation applied to all layers) and a partial Mantel test
(no maintained Python implementation exists; note that partial Mantel tests can
have inflated type-I error under autocorrelation, Guillot & Rousset 2013,
https://doi.org/10.1111/2041-210X.12018, so they are used only as secondary
evidence).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata


def upper(D: np.ndarray) -> np.ndarray:
    """Strict upper-triangle entries of [n, n] (or [..., n, n]) as a vector."""
    n = D.shape[-1]
    i, j = np.triu_indices(n, 1)
    return D[..., i, j]


def _z(x: np.ndarray) -> np.ndarray:
    """Standardize along the last axis (mean 0, std 1)."""
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x**2).mean(-1, keepdims=True))


def _transform(v: np.ndarray, method: str) -> np.ndarray:
    """Rank-transform (Spearman) or leave as is (Pearson), then standardize."""
    if method == "spearman":
        v = np.apply_along_axis(rankdata, -1, v) if v.ndim > 1 else rankdata(v)
    elif method != "pearson":
        raise ValueError(method)
    return _z(v.astype(float))


def mantel_r(D1: np.ndarray, D2: np.ndarray, method: str = "spearman") -> np.ndarray | float:
    """Mantel correlation of D1 ([n, n] or [L, n, n]) with D2 ([n, n])."""
    a, b = _transform(upper(D1), method), _transform(upper(D2), method)
    return (a * b).mean(-1)


def _perm_matrix_upper(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Upper triangle of R with rows and columns permuted by p."""
    return upper(R[np.ix_(p, p)])


def mantel_maxT(D_layers: np.ndarray, D_ref: np.ndarray, n_perm: int, rng: np.random.Generator,
                method: str = "spearman", groups: np.ndarray | None = None):
    """Mantel r per layer and a max-T permutation p-value across layers.

    Local: each layer's upper triangle is transformed once; the reference matrix
    is permuted (rows/cols together; optionally only within ``groups``). Because
    permuting rows/cols maps the upper-triangle multiset to itself, ranks of the
    permuted reference equal the permuted ranks, so ranks are computed once.
    Returns (r per layer [L], p_maxT, null maxima [n_perm]).
    """
    L = _transform(upper(np.atleast_3d(D_layers) if D_layers.ndim == 3 else D_layers[None]), method)
    n = D_ref.shape[0]
    # Matrix of transformed reference values (symmetric), for fast permutation.
    ref_t = np.zeros_like(D_ref, dtype=float)
    i, j = np.triu_indices(n, 1)
    ref_t[i, j] = _transform(upper(D_ref), method)
    ref_t = ref_t + ref_t.T
    obs = (L * upper(ref_t)).mean(-1)
    null = np.empty(n_perm)
    g = np.zeros(n, int) if groups is None else np.asarray(groups)
    for k in range(n_perm):
        p = np.arange(n)
        for gv in np.unique(g):
            idx = np.where(g == gv)[0]
            p[idx] = rng.permutation(idx)
        null[k] = (L * _perm_matrix_upper(ref_t, p)).mean(-1).max()
    pval = (1 + np.sum(null >= obs.max())) / (n_perm + 1)
    return obs, float(pval), null


def partial_mantel(D1: np.ndarray, D2: np.ndarray, D3: np.ndarray, n_perm: int,
                   rng: np.random.Generator, method: str = "spearman"):
    """Partial Mantel correlation r(D1, D2 | D3) with a permutation p-value (one-sided, > 0).

    Local: partial correlation of the (rank-)transformed upper triangles,
    r12.3 = (r12 - r13 r23) / sqrt((1 - r13^2)(1 - r23^2)); the null permutes D1's
    rows/columns (as in vegan::mantel.partial).
    """
    a, b, c = (_transform(upper(D), method) for D in (D1, D2, D3))

    def pr(x):
        r12, r13, r23 = (x * b).mean(), (x * c).mean(), (b * c).mean()
        return (r12 - r13 * r23) / np.sqrt((1 - r13**2) * (1 - r23**2))

    obs = pr(a)
    n = D1.shape[0]
    A = np.zeros_like(D1, dtype=float)
    i, j = np.triu_indices(n, 1)
    A[i, j] = a
    A = A + A.T
    null = np.array([pr(_perm_matrix_upper(A, rng.permutation(n))) for _ in range(n_perm)])
    return float(obs), float((1 + np.sum(null >= obs)) / (n_perm + 1))


def kendall_tau_b(D1: np.ndarray, D2: np.ndarray) -> float:
    """Kendall tau-b between upper triangles (robust to the many ties of tree distances)."""
    return float(kendalltau(upper(D1), upper(D2), variant="b").statistic)
