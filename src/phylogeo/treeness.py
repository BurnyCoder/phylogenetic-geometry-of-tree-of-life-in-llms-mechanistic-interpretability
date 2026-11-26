"""Tree-likeness of a distance matrix: Gromov delta-hyperbolicity and neighbor-joining fit.

Global context (stage 2, H_geom): is the representation geometry tree-shaped?
  * Gromov four-point delta: a metric is a tree metric iff delta = 0. We use the
    base-point formulation delta_w = max_{x,y,z} min((x|z)_w, (y|z)_w) - (x|y)_w
    with Gromov products (x|y)_w = (d(x,w) + d(y,w) - d(x,y)) / 2, computed exactly
    by a (max, min) matrix product, and report delta_rel = 2 delta_w / diam as in
    Khrulkov et al. 2020 (https://arxiv.org/abs/1904.02239), maximized over a few
    base points. delta_rel is near 0 for trees and grows for non-tree metrics.
    High-dimensional noise also looks tree-like, so delta must be compared with a
    spectrum-matched Gaussian null (see literature-review.md section 3).
  * Neighbor joining (Saitou & Nei 1987) on the representation distances via
    scikit-bio, compared with the reference topology by normalized Robinson-Foulds
    distance (DendroPy), against NJ trees from label-permuted matrices.
"""

from __future__ import annotations

import numpy as np


def gromov_products(D: np.ndarray, w: int) -> np.ndarray:
    """[n, n] Gromov products (x|y)_w."""
    return 0.5 * (D[:, w][:, None] + D[w, :][None, :] - D)


def maxmin_product(A: np.ndarray, B: np.ndarray, chunk: int = 64) -> np.ndarray:
    """(A (max,min) B)[i, j] = max_k min(A[i, k], B[k, j]), chunked to bound memory."""
    n = A.shape[0]
    out = np.empty((n, B.shape[1]))
    for s in range(0, n, chunk):
        out[s : s + chunk] = np.minimum(A[s : s + chunk, :, None], B[None, :, :]).max(axis=1)
    return out


def delta_base_point(D: np.ndarray, w: int) -> float:
    """Exact four-point delta with respect to base point w (O(n^3))."""
    G = gromov_products(D, w)
    return float((maxmin_product(G, G) - G).max())


def delta_rel(D: np.ndarray, n_base: int = 8, rng: np.random.Generator | None = None) -> float:
    """2 * max_w delta_w / diam over ``n_base`` random base points (Khrulkov et al. 2020)."""
    rng = rng or np.random.default_rng(0)
    ws = rng.choice(D.shape[0], size=min(n_base, D.shape[0]), replace=False)
    return 2.0 * max(delta_base_point(D, int(w)) for w in ws) / D.max()


def delta_bruteforce(D: np.ndarray) -> float:
    """Four-point delta by enumerating all quadruples (tests only; O(n^4))."""
    n = D.shape[0]
    best = 0.0
    import itertools

    for x, y, z, w in itertools.combinations(range(n), 4):
        s = sorted([D[x, y] + D[z, w], D[x, z] + D[y, w], D[x, w] + D[y, z]], reverse=True)
        best = max(best, (s[0] - s[1]) / 2)
    return best


def nj_newick(D: np.ndarray, labels: list[str]) -> str:
    """Neighbor-joining tree (scikit-bio) of distance matrix D as a Newick string."""
    from skbio import DistanceMatrix
    from skbio.tree import nj

    Dm = (D + D.T) / 2
    np.fill_diagonal(Dm, 0.0)
    return str(nj(DistanceMatrix(Dm, labels))).strip()  # TreeNode.__str__ -> Newick


def rf_normalized(newick_a: str, newick_b: str) -> float:
    """Unrooted RF divided by the binary-tree upper bound; unresolved reference splits reduce the attainable maximum."""
    import dendropy
    from dendropy.calculate import treecompare

    tns = dendropy.TaxonNamespace()
    ta = dendropy.Tree.get(data=newick_a, schema="newick", taxon_namespace=tns, preserve_underscores=True)
    tb = dendropy.Tree.get(data=newick_b, schema="newick", taxon_namespace=tns, preserve_underscores=True)
    for t in (ta, tb):
        t.is_rooted = False
        t.encode_bipartitions()
    rf = treecompare.symmetric_difference(ta, tb)
    n = len(tns)
    # Max RF for two unrooted binary trees = 2 (n - 3); polytomies lower the count of splits.
    return rf / (2 * (n - 3))
