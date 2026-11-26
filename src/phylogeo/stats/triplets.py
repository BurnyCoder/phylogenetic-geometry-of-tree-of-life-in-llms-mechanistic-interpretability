"""Rooted-triplet accuracy (TA): the preregistered primary metric.

Global context: for species (a, b, c) such that the reference tree puts a and b
closer (MRCA(a,b) strictly below MRCA(a,b,c)), a representation "gets the
triplet right" if d(a,b) < min(d(a,c), d(b,c)). Chance = 1/3. TA needs only the
tree topology, is invariant to monotone transforms of distance, and, with
node-balanced sampling, gives equal weight to every branching point of the tree
(so big clades cannot dominate). The triplet idea is the basis of the rooted
triplet distance between trees (Critchlow, Pearl & Qian 1996,
https://doi.org/10.1093/sysbio/45.3.323). This module also holds the
inference machinery for H1: max-T permutation over layers (Westfall & Young
1993), family-blocked cross-fitting for layer choice, and the clade-block
bootstrap (preregistration section 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phylogeo.data.tree import RefTree


@dataclass
class Triplets:
    """Arrays of resolved triplets: (a, b) is the tree-closer pair, c the outgroup."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    node: np.ndarray  # decision node = MRCA(a, b, c)

    def subset(self, m: np.ndarray) -> "Triplets":
        return Triplets(self.a[m], self.b[m], self.c[m], self.node[m])

    def __len__(self) -> int:
        return len(self.a)


def sample_triplets(tree: RefTree, k_per_node: int, rng: np.random.Generator) -> Triplets:
    """Sample up to ``k_per_node`` resolved triplets uniformly at every internal node.

    Local: at node v with children C_1..C_m, a resolved triplet has a, b in the
    same child C_i (a != b) and c in another child. There are
    T_v = sum_i C(|C_i|, 2) * (|v| - |C_i|) of them; if T_v <= k all are
    enumerated, otherwise k are drawn uniformly (child chosen with probability
    proportional to its share of T_v, then a random pair inside and a random c
    outside). Triplets spanning three different children of a polytomy are
    unresolved and never produced.
    """
    A, B, C, V = [], [], [], []
    for v in tree.internal_nodes:
        ch = [tree.members[c] for c in tree.children[v]]
        sizes = np.array([len(x) for x in ch])
        n_v = sizes.sum()
        w = sizes * (sizes - 1) / 2 * (n_v - sizes)
        total = int(w.sum())
        if total == 0:
            continue
        if total <= k_per_node:  # enumerate exhaustively
            for i, ci in enumerate(ch):
                if len(ci) < 2:
                    continue
                out = np.concatenate([x for j, x in enumerate(ch) if j != i])
                for p in range(len(ci)):
                    for q in range(p + 1, len(ci)):
                        for c in out:
                            A.append(ci[p]); B.append(ci[q]); C.append(c); V.append(v)
        else:  # uniform sampling
            picks = rng.choice(len(ch), size=k_per_node, p=w / w.sum())
            for i in picks:
                ci = ch[i]
                p, q = rng.choice(len(ci), size=2, replace=False)
                out = np.concatenate([x for j, x in enumerate(ch) if j != i])
                A.append(ci[p]); B.append(ci[q]); C.append(int(rng.choice(out))); V.append(v)
    return Triplets(np.array(A, int), np.array(B, int), np.array(C, int), np.array(V, int))


def triplet_scores(D: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Correctness of each triplet under distance matrix D (or a stack [L, n, n]).

    Section 4.1 of the registration assigns one-third credit to a tied minimum.
    The historical shared-minimum rule is available in verification.statistics for replay.
    """
    dab, dac, dbc = D[..., a, b], D[..., a, c], D[..., b, c]
    m = np.minimum(np.minimum(dab, dac), dbc)
    n_min = (dab == m).astype(float) + (dac == m) + (dbc == m)
    return np.where(n_min > 1, 1 / 3, (dab == m).astype(float))


def node_balanced_mean(scores: np.ndarray, node: np.ndarray, mask: np.ndarray | None = None,
                       weights: np.ndarray | None = None,
                       node_mult: dict[int, float] | None = None) -> np.ndarray | float:
    """Mean over decision nodes of the (weighted) mean triplet score at that node.

    ``scores`` may be [T] or [L, T]. ``weights`` (per triplet) and ``node_mult``
    (per node) are used by the clade-block bootstrap. Nodes with no (positive
    weight) triplet in ``mask`` are skipped. Returns NaN if no triplet qualifies.
    """
    T = scores.shape[-1]
    keep = np.ones(T, bool) if mask is None else mask.astype(bool)
    w = np.ones(T) if weights is None else weights.astype(float)
    keep = keep & (w > 0)
    if not keep.any():
        return np.full(scores.shape[:-1], np.nan) if scores.ndim > 1 else float("nan")
    nodes, inv = np.unique(node[keep], return_inverse=True)
    sw = w[keep]
    num = np.zeros(scores.shape[:-1] + (len(nodes),))
    den = np.bincount(inv, weights=sw, minlength=len(nodes))
    s = scores[..., keep] * sw
    if scores.ndim == 1:
        num = np.bincount(inv, weights=s, minlength=len(nodes))
    else:
        for l in range(scores.shape[0]):
            num[l] = np.bincount(inv, weights=s[l], minlength=len(nodes))
    acc = num / den
    m = np.ones(len(nodes)) if node_mult is None else np.array([node_mult.get(int(v), 1.0) for v in nodes])
    return (acc * m).sum(-1) / m.sum()


def ta(D: np.ndarray, trip: Triplets, mask: np.ndarray | None = None) -> np.ndarray | float:
    """Node-balanced triplet accuracy of D ([n, n] or [L, n, n]) on ``trip``."""
    return node_balanced_mean(triplet_scores(D, trip.a, trip.b, trip.c), trip.node, mask)


def within_group_mask(trip: Triplets, groups: np.ndarray, exclude: tuple = ("other",)) -> np.ndarray:
    """True where a, b, c share a group label (e.g. stratum) not in ``exclude``."""
    g = np.asarray(groups)
    m = (g[trip.a] == g[trip.b]) & (g[trip.b] == g[trip.c])
    return m & ~np.isin(g[trip.a], list(exclude))


def distinct_mask(trip: Triplets, labels: np.ndarray) -> np.ndarray:
    """True where the three labels (e.g. head nouns) are pairwise different."""
    h = np.asarray(labels)
    return (h[trip.a] != h[trip.b]) & (h[trip.a] != h[trip.c]) & (h[trip.b] != h[trip.c])


def discordant_mask(B: np.ndarray, trip: Triplets) -> np.ndarray:
    """True where exactly one alternative to the sister pair is strictly closest."""
    return triplet_scores(B, trip.a, trip.b, trip.c) == 0.0


def shared_rank_level(a, b, c, ranks: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """Deepest rank (in ``order``, coarse -> fine) shared by all three species, else "none".

    Empty rank names never count as shared.
    """
    out = np.full(len(a), "none", dtype=object)
    for r in order:  # coarse to fine; finer shared ranks overwrite
        v = np.asarray(ranks[r])
        same = (v[a] == v[b]) & (v[b] == v[c]) & (v[a] != "")
        out[same] = r
    return out


def _group_permutation(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of indices that only swaps elements within the same group."""
    perm = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def maxT_ta(D_layers: np.ndarray, trip: Triplets, mask: np.ndarray | None, groups: np.ndarray,
            n_perm: int, rng: np.random.Generator):
    """Max-T permutation test of TA across layers.

    Local: the observed statistic is TA at each layer; for each permutation the
    representation-to-leaf assignment is shuffled *within groups* (same
    permutation for all layers) and the maximum TA over layers is recorded.
    p = (1 + #{max_perm >= max_obs}) / (n_perm + 1). This controls the
    family-wise error over layers (Westfall & Young 1993).
    Returns (observed TA per layer [L], p_maxT, null maxima [n_perm]).
    """
    obs = np.atleast_1d(ta(D_layers, trip, mask))
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = _group_permutation(np.asarray(groups), rng)
        s = triplet_scores(D_layers, p[trip.a], p[trip.b], p[trip.c])
        null[i] = np.nanmax(np.atleast_1d(node_balanced_mean(s, trip.node, mask)))
    pval = (1 + np.sum(null >= np.nanmax(obs))) / (n_perm + 1)
    return obs, float(pval), null


def crossfit_ta(D_layers: np.ndarray, trip: Triplets, mask: np.ndarray | None, families: np.ndarray,
                n_splits: int, rng: np.random.Generator):
    """Family-blocked 2-fold cross-fitting of the layer choice.

    Local: families are split at random into halves A/B; the best layer is chosen
    on triplets whose three species all lie in A and evaluated on triplets fully
    in B, then vice versa; the two values are averaged. Repeated ``n_splits``
    times. Returns (mean cross-fitted TA, array of chosen layers).
    """
    fam = np.asarray(families)
    ufam = np.unique(fam)
    base = np.ones(len(trip), bool) if mask is None else mask
    vals, chosen = [], []
    for _ in range(n_splits):
        half = set(rng.permutation(ufam)[: len(ufam) // 2])
        inA = np.isin(fam, list(half))
        mA = base & inA[trip.a] & inA[trip.b] & inA[trip.c]
        mB = base & ~inA[trip.a] & ~inA[trip.b] & ~inA[trip.c]
        split_vals = []
        for fit, ev in ((mA, mB), (mB, mA)):
            if not fit.any() or not ev.any():
                continue
            l_best = int(np.nanargmax(np.atleast_1d(ta(D_layers, trip, fit))))
            chosen.append(l_best)
            split_vals.append(float(ta(D_layers[l_best], trip, ev)))
        if split_vals:
            vals.append(np.mean(split_vals))
    return float(np.mean(vals)), np.array(chosen, int)


def clade_bootstrap_ta(Ds: list[np.ndarray], trip: Triplets, mask: np.ndarray | None,
                       families: np.ndarray, tree: RefTree, n_boot: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Clade-block (family) bootstrap distribution of TA for several distance matrices.

    Local: families are resampled with replacement; species weight = number of
    times its family was drawn; triplet weight = product of its species' weights
    (triplets with two copies of one species do not exist because only distinct
    species are ever combined). Nodes inside a single family are counted with
    multiplicity = that family's count; nodes spanning families count once. The
    same draws are used for every matrix in ``Ds`` so differences are paired.
    Returns [len(Ds), n_boot].
    """
    fam = np.asarray(families)
    ufam, fam_idx = np.unique(fam, return_inverse=True)
    # Which decision nodes lie inside a single family (and which family)?
    node_fam = {}
    for v in np.unique(trip.node):
        fs = np.unique(fam_idx[tree.members[v]])
        node_fam[int(v)] = int(fs[0]) if len(fs) == 1 else -1
    scores = [triplet_scores(D, trip.a, trip.b, trip.c) for D in Ds]
    out = np.empty((len(Ds), n_boot))
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, len(ufam), len(ufam)), minlength=len(ufam)).astype(float)
        w_sp = counts[fam_idx]
        w = w_sp[trip.a] * w_sp[trip.b] * w_sp[trip.c]
        mult = {v: (counts[f] if f >= 0 else 1.0) for v, f in node_fam.items()}
        for j, s in enumerate(scores):
            out[j, i] = node_balanced_mean(s, trip.node, mask, weights=w, node_mult=mult)
    return out
