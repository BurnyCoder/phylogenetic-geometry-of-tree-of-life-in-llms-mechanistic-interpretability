"""Synthetic point clouds on the frozen reference tree (positive/negative controls, power).

Global context:
  * Brownian motion along the tree (each edge adds an isotropic Gaussian step) is
    the textbook model of trait evolution on a phylogeny (Felsenstein 1985,
    https://doi.org/10.1086/284325): a "perfect tree-structured" representation.
  * The class-blob null puts each species at its stratum centroid plus isotropic
    noise: clusters for folk classes but no structure inside them (H_folk).
  * The power analysis mixes the two at varying strength and runs the exact H1
    test machinery (preregistration section 4.5).
"""

from __future__ import annotations

import logging

import numpy as np

from phylogeo.data.tree import RefTree

log = logging.getLogger("phylogeo.simulate")


def brownian_tree(tree: RefTree, dim: int, rng: np.random.Generator) -> np.ndarray:
    """[n, dim] leaf values of unit-rate Brownian motion along every edge (unit edge length)."""
    X = np.zeros((tree.N, dim))
    for v in tree.preorder:
        for c in tree.children[v]:
            X[c] = X[v] + rng.normal(size=dim)
    return X[: tree.n]


def class_blob(strata: np.ndarray, dim: int, rng: np.random.Generator, spread: float = 1.0) -> np.ndarray:
    """[n, dim] stratum centroid (random, unit scale) + isotropic noise of scale ``spread``."""
    s = np.asarray(strata)
    cents = {g: rng.normal(size=dim) for g in np.unique(s)}
    return np.stack([cents[g] for g in s]) + spread * rng.normal(size=(len(s), dim))


def mixture(tree: RefTree, strata: np.ndarray, signal: float, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Standardized class-blob + ``signal`` x standardized Brownian-tree component."""
    B = brownian_tree(tree, dim, rng)
    C = class_blob(strata, dim, rng)
    B = (B - B.mean(0)) / B.std()
    C = (C - C.mean(0)) / C.std()
    return C + signal * B


def power_curve(tree: RefTree, strata: np.ndarray, heads: np.ndarray, signals: list[float],
                n_sims: int, n_perm: int, k_per_node: int, dim: int, seed: int) -> list[dict]:
    """Power of the H1 test (within-stratum TA, within-stratum permutation) vs signal strength.

    Single-layer version of the preregistered test (max-T over 25 layers is more
    conservative; the minimum detectable effect reported is therefore optimistic
    by a small margin, which is stated in the report).
    """
    from phylogeo.geometry import cosine_distance
    from phylogeo.stats.triplets import distinct_mask, maxT_ta, sample_triplets, within_group_mask

    rng = np.random.default_rng(seed)
    trip = sample_triplets(tree, k_per_node, rng)
    mask = within_group_mask(trip, strata) & distinct_mask(trip, heads)
    out = []
    for s in signals:
        tas, ps = [], []
        for _ in range(n_sims):
            X = mixture(tree, strata, s, dim, rng)
            D = cosine_distance(X - X.mean(0))
            obs, p, _ = maxT_ta(D[None], trip, mask, strata, n_perm, rng)
            tas.append(float(obs[0]))
            ps.append(p)
        rec = {"signal": s, "ta_within_mean": float(np.mean(tas)), "ta_within_sd": float(np.std(tas)),
               "power": float(np.mean(np.array(ps) < 0.05))}
        log.info("power sim: %s", rec)
        out.append(rec)
    return out
