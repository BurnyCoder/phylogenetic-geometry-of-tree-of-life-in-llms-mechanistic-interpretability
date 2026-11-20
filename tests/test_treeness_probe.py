"""Ground-truth tests for delta-hyperbolicity, NJ/RF, and the flat probe."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from phylogeo.probes.flat import cv_probe, family_folds
from phylogeo.treeness import delta_base_point, delta_bruteforce, delta_rel, nj_newick, rf_normalized
from tests.conftest import brownian_on_tree


def test_delta_zero_on_tree_metric(tree30):
    D = tree30.path_length()  # additive tree metric (unit edges)
    assert delta_rel(D) == pytest.approx(0.0, abs=1e-9)


def test_delta_cycle_graph_known_value():
    # Cycle C_n with n = 4m: four-point delta = m/... -> compare to brute force exactly.
    n = 12
    i = np.arange(n)
    D = np.minimum(np.abs(i[:, None] - i[None]), n - np.abs(i[:, None] - i[None])).astype(float)
    bf = delta_bruteforce(D)
    assert bf > 0
    # Base-point delta bounds the four-point delta: delta_4pt <= delta_w <= 2 * delta_4pt... check max over w.
    dw = max(delta_base_point(D, w) for w in range(n))
    assert bf <= dw + 1e-9 and dw <= 2 * bf + 1e-9


def test_delta_base_point_matches_bruteforce_on_random(rng):
    X = rng.normal(size=(15, 3))
    D = cdist(X, X)
    bf = delta_bruteforce(D)
    dw = max(delta_base_point(D, w) for w in range(15))
    assert bf <= dw + 1e-9 <= 2 * bf + 1e-9


def test_nj_recovers_additive_tree(tree30):
    D = tree30.path_length()
    nwk = nj_newick(D, tree30.leaf_labels)
    assert rf_normalized(nwk, tree30.to_newick()) == pytest.approx(0.0)


def test_rf_random_tree_is_large(tree30, rng):
    X = rng.normal(size=(30, 5))
    nwk = nj_newick(cdist(X, X), tree30.leaf_labels)
    assert rf_normalized(nwk, tree30.to_newick()) > 0.6


def test_family_folds_keep_families_together(rng):
    fam = np.repeat(np.arange(20), 3)
    folds = family_folds(fam, 5, rng)
    for m in folds:
        for f in np.unique(fam[m]):
            assert m[fam == f].all()
    assert np.sum([m.sum() for m in folds]) == 60


def test_probe_recovers_planted_tree_and_control_fails(tree30, rng):
    X = brownian_on_tree(tree30, 40, rng)
    V = np.hstack([X, rng.normal(size=(30, 60))])  # tree signal + nuisance dims
    D = tree30.path_length()
    fam = np.arange(30) // 2
    r = cv_probe(V, D, fam, k=5, n_folds=3, n_pca=20, steps=400, seed=0)
    perm = rng.permutation(30)
    rc = cv_probe(V, D[np.ix_(perm, perm)], fam, k=5, n_folds=3, n_pca=20, steps=400, seed=0)
    assert r["test_test_pearson"][0] > 0.4
    assert r["test_test_pearson"][0] > rc["test_test_pearson"][0] + 0.2
