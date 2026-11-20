"""Ground-truth tests for rooted-triplet accuracy (the preregistered primary metric)."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from phylogeo.data.tree import RefTree
from phylogeo.stats.triplets import (
    clade_bootstrap_ta,
    crossfit_ta,
    discordant_mask,
    maxT_ta,
    node_balanced_mean,
    sample_triplets,
    shared_rank_level,
    ta,
    triplet_scores,
)
from tests.conftest import brownian_on_tree


def test_exact_tree_metric_gives_ta_one(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=50, rng=rng)
    assert len(trip.a) > 0
    assert ta(tree30.lca_depth_distance(), trip) == pytest.approx(1.0)


def test_triplets_are_resolved_by_tree(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=50, rng=rng)
    L = tree30.lca
    d = tree30.depth
    # MRCA(a,b) strictly below MRCA(a,b,c) = node.
    assert np.all(d[L[trip.a, trip.b]] > d[trip.node])
    assert np.all(L[trip.a, trip.c] == trip.node) and np.all(L[trip.b, trip.c] == trip.node)


def test_random_vectors_give_chance(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=200, rng=rng)
    vals = []
    for _ in range(30):
        X = rng.normal(size=(30, 64))
        vals.append(ta(cdist(X, X), trip))
    assert np.mean(vals) == pytest.approx(1 / 3, abs=0.03)


def test_brownian_tree_cloud_beats_chance(tree30, rng):
    X = brownian_on_tree(tree30, dim=256, rng=rng)
    trip = sample_triplets(tree30, k_per_node=200, rng=rng)
    assert ta(cdist(X, X), trip) > 0.7


def test_polytomy_triplets_excluded():
    t = RefTree("((A,B,C),D);", ["A", "B", "C", "D"])
    trip = sample_triplets(t, k_per_node=1000, rng=np.random.default_rng(1))
    # The only resolved triplets have the pair inside {A,B,C} and c = D.
    assert set(trip.c.tolist()) == {3}
    assert len(trip.a) > 0


def test_tie_handling():
    a, b, c = np.array([0]), np.array([1]), np.array([2])
    D = np.zeros((3, 3))
    assert triplet_scores(D, a, b, c)[0] == pytest.approx(1 / 3)
    D2 = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], float)  # d_ab = d_ac < d_bc
    assert triplet_scores(D2, a, b, c)[0] == pytest.approx(1 / 2)
    D3 = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]], float)
    assert triplet_scores(D3, a, b, c)[0] == 1.0
    D4 = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], float)
    assert triplet_scores(D4, a, b, c)[0] == 0.0


def test_node_balanced_mean():
    scores = np.array([1, 1, 1, 1, 0.0])
    node = np.array([7, 7, 7, 7, 9])
    assert node_balanced_mean(scores, node) == pytest.approx(0.5)  # not 0.8
    mask = np.array([True, True, True, True, False])
    assert node_balanced_mean(scores, node, mask) == pytest.approx(1.0)


def test_layers_vectorized(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=50, rng=rng)
    D = np.stack([tree30.lca_depth_distance(), cdist(X := rng.normal(size=(30, 8)), X)])
    v = ta(D, trip)
    assert v.shape == (2,) and v[0] == pytest.approx(1.0)


def test_discordant_mask(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=50, rng=rng)
    assert not discordant_mask(tree30.lca_depth_distance(), trip).any()  # perfect baseline never wrong
    m = discordant_mask(-tree30.lca_depth_distance(), trip)  # anti-tree baseline is always wrong
    assert m.all()


def test_maxT_null_and_signal(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=100, rng=rng)
    groups = np.zeros(30, dtype=int)
    # Null: random distances in 3 "layers" -> p should not be tiny.
    ps = []
    for s in range(20):
        r = np.random.default_rng(100 + s)
        D = np.stack([cdist(X := r.normal(size=(30, 8)), X) for _ in range(3)])
        ps.append(maxT_ta(D, trip, None, groups, n_perm=99, rng=r)[1])
    assert np.mean(np.array(ps) < 0.05) <= 0.2
    # Signal in one of 3 layers -> small p.
    X = brownian_on_tree(tree30, 64, rng)
    D = np.stack([cdist(Y := rng.normal(size=(30, 8)), Y), cdist(X, X), cdist(Y, Y)])
    obs, p, _ = maxT_ta(D, trip, None, groups, n_perm=199, rng=rng)
    assert p < 0.01 and obs.argmax() == 1


def test_crossfit_picks_signal_layer(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=100, rng=rng)
    X = brownian_on_tree(tree30, 64, rng)
    Y = rng.normal(size=(30, 8))
    D = np.stack([cdist(Y, Y), cdist(X, X)])
    fam = np.arange(30) // 3  # 10 "families" of 3 consecutive leaves
    val, layers = crossfit_ta(D, trip, None, fam, n_splits=5, rng=rng)
    assert val > 0.6 and np.bincount(layers).argmax() == 1


def test_bootstrap_identity_and_spread(tree30, rng):
    trip = sample_triplets(tree30, k_per_node=100, rng=rng)
    X = brownian_on_tree(tree30, 16, rng, noise=3.0)
    D = cdist(X, X)
    fam = np.arange(30) // 3
    boots = clade_bootstrap_ta([D], trip, None, fam, tree30, n_boot=200, rng=rng)
    assert boots.shape == (1, 200)
    point = ta(D, trip)
    assert boots.min() < point < boots.max()  # point estimate inside bootstrap spread
    assert abs(np.median(boots) - point) < 0.1


def test_shared_rank_level():
    ranks = {"order": np.array(["O1", "O1", "O1", "O2"]), "family": np.array(["F1", "F1", "F2", "F3"])}
    a, b, c = np.array([0, 0]), np.array([1, 1]), np.array([2, 3])
    lv = shared_rank_level(a, b, c, ranks, ["order", "family"])
    assert list(lv) == ["order", "none"]


def test_path_length_is_not_triplet_consistent(tree30, rng):
    """Documents why S1's edge-count distance differs from the triplet oracle."""
    trip = sample_triplets(tree30, k_per_node=50, rng=rng)
    assert ta(tree30.path_length(), trip) < 1.0
