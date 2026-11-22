"""The audit's independent triplet enumerator must agree with the main sampler when nothing is capped.

Global context: phase ``audit`` re-derives TA_within with code that shares nothing with
``stats/triplets.py`` (dendropy LCAs, full enumeration). If both are correct they give the
same node-balanced accuracy when the sampler enumerates every triplet (k_per_node large).
"""

import numpy as np

from phylogeo.audit import exhaustive_within_triplets, node_balanced_ta
from phylogeo.stats.triplets import distinct_mask, sample_triplets, ta, within_group_mask


def _setup(tree30, rng):
    groups = np.array(["g1"] * 15 + ["g2"] * 10 + ["other"] * 5)
    heads = np.array([f"h{i % 12}" for i in range(30)])  # some shared head nouns
    return groups, heads


def test_exhaustive_matches_uncapped_sampler(tree30, rng):
    groups, heads = _setup(tree30, rng)
    D = rng.random((30, 30))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    a, b, c, v = exhaustive_within_triplets(tree30.to_newick(), tree30.leaf_labels, groups, heads)
    trip = sample_triplets(tree30, 10**9, rng)  # cap never reached: full enumeration
    m = within_group_mask(trip, groups) & distinct_mask(trip, heads)
    assert len(a) == int(m.sum())
    assert np.isclose(node_balanced_ta(D, a, b, c, v), ta(D, trip, m))


def test_exhaustive_oracles(tree30, rng):
    groups, heads = _setup(tree30, rng)
    a, b, c, v = exhaustive_within_triplets(tree30.to_newick(), tree30.leaf_labels, groups, heads)
    assert np.isclose(node_balanced_ta(tree30.lca_depth_distance(), a, b, c, v), 1.0)  # the tree itself
    assert np.isclose(node_balanced_ta(np.ones((30, 30)), a, b, c, v), 1 / 3)  # all ties: chance
