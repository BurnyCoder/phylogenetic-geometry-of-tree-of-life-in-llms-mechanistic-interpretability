"""Ground-truth tests for geometry helpers and stage-1a PCA checks."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from phylogeo.geometry import (
    cosine_distance,
    knn_geodesic,
    smallest_connected_k,
    species_vectors,
)
from phylogeo.data.tree import RefTree
from phylogeo.viz.pca import (
    fit_tree_positions,
    fitted_sq_length,
    knn_purity,
    overlay_length_test,
    overlay_positions,
    pca_top3,
    tree_fit_test,
)
from tests.conftest import brownian_on_tree


def test_species_vectors_centering():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4, 3, 8)) + np.arange(4)[None, :, None, None] * 100  # template offsets
    V = species_vectors(X)
    assert V.shape == (3, 10, 8)
    assert np.allclose(V.mean(1), 0, atol=1e-10)  # centered over species
    Vr = species_vectors(X, center=False)
    assert not np.allclose(Vr.mean(1), 0)


def test_cosine_distance_basic():
    V = np.array([[1.0, 0], [0, 1], [-1, 0]])
    D = cosine_distance(V)
    assert D[0, 1] == pytest.approx(1) and D[0, 2] == pytest.approx(2) and D[0, 0] == 0


def test_geodesic_recovers_circle_arclength():
    rng = np.random.default_rng(0)
    # Evenly spaced angles with small jitter: no single large gap that the
    # smallest-connected-k graph could leave unbridged (see next test).
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False) + 0.005 * rng.normal(size=120)
    X = np.c_[np.cos(th), np.sin(th)] + 0.002 * rng.normal(size=(120, 2))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    G, k = knn_geodesic(D)
    dth = np.abs(th[:, None] - th[None])
    arc = np.minimum(dth, 2 * np.pi - dth)
    iu = np.triu_indices(120, 1)
    assert spearmanr(G[iu], arc[iu]).statistic > 0.99


def test_smallest_connected_k_can_tear_a_cycle():
    """Documented limitation: with one big gap the minimal connected graph is an open chain."""
    th = np.r_[np.linspace(0, 1.5 * np.pi, 60)]  # 3/4 circle: big gap between ends
    X = np.c_[np.cos(th), np.sin(th)]
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    G, k = knn_geodesic(D)
    # Endpoints are close in the plane (chord ~1.41) but far along the chain (~4.7).
    assert G[0, -1] > 3 * D[0, -1]


def test_smallest_connected_k_two_blobs():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(size=(20, 2)), rng.normal(size=(20, 2)) + 100]
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    # Two far blobs of 20 points: connected only when k reaches 20.
    assert smallest_connected_k(D) == 20


def test_pca_recovers_circle_order_in_high_dim():
    rng = np.random.default_rng(1)
    th = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    basis = np.linalg.qr(rng.normal(size=(2048, 2)))[0]
    X = (np.c_[np.cos(th), np.sin(th)] * 10) @ basis.T + 0.1 * rng.normal(size=(7, 2048))
    c, evr, _ = pca_top3(X)
    ang = np.unwrap(np.arctan2(c[:, 1], c[:, 0]))
    d = np.diff(ang)
    assert np.all(d > 0) or np.all(d < 0)  # cyclic order preserved


def test_overlay_positions_are_centroids(tree30):
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(30, 3))
    pos = overlay_positions(tree30, coords)
    for v in tree30.internal_nodes:
        assert np.allclose(pos[v], coords[tree30.members[v]].mean(0))
    assert np.allclose(pos[:30], coords)


def test_overlay_length_detects_tree_signal(tree30):
    rng = np.random.default_rng(0)
    X = brownian_on_tree(tree30, 3, rng)
    _, _, p = overlay_length_test(tree30, X, n_perm=199, rng=rng)
    assert p < 0.01
    Z = rng.normal(size=(30, 3))
    _, _, p0 = overlay_length_test(tree30, Z, n_perm=199, rng=rng)
    assert p0 > 0.01


def test_fitted_tree_star_centre_is_the_leaf_mean():
    """Star tree: minimizing sum ||x_leaf - x_centre||^2 puts the centre at the mean; leaves stay put."""
    t = RefTree("(a,b,c,d);", list("abcd"))
    X = np.random.default_rng(0).normal(size=(4, 3))
    pos = fit_tree_positions(t, X)
    assert np.allclose(pos[t.root], X.mean(0)) and np.allclose(pos[:4], X)


def test_fitted_tree_matches_hand_solution_on_a_chain():
    """((a,b),c), unit edges: d/du and d/dr of (u-a)^2+(u-b)^2+(r-u)^2+(r-c)^2 give u = (2a+2b+c)/5 and
    r = (a+b+3c)/5; with a = b = 0, c = 5: u = 1, r = 3 (not the centroids 0 and 5/3)."""
    t = RefTree("((a,b),c);", list("abc"))
    pos = fit_tree_positions(t, np.array([[0.0], [0.0], [5.0]]))
    u = t.parent[0]
    assert pos[u, 0] == pytest.approx(1.0) and pos[t.root, 0] == pytest.approx(3.0)


def test_fitted_tree_is_never_longer_than_the_centroid_overlay(tree30):
    """The fit is the least-squares placement, so its squared edge length is <= that of any other placement."""
    X = np.random.default_rng(1).normal(size=(30, 3))
    cen = overlay_positions(tree30, X)
    child = np.arange(tree30.N) != tree30.root
    sq_cen = float(((cen[child] - cen[tree30.parent[child]]) ** 2).sum())
    assert fitted_sq_length(tree30, X) <= sq_cen + 1e-9


def test_tree_fit_ratio_detects_tree_signal_and_not_noise(tree30):
    """Brownian motion on the tree fits far better than shuffled labels (ratio << 1); i.i.d. noise does not."""
    rng = np.random.default_rng(0)
    ratio, p = tree_fit_test(tree30, brownian_on_tree(tree30, 3, rng), 199, rng)
    assert ratio < 0.7 and p < 0.01
    ratio0, p0 = tree_fit_test(tree30, rng.normal(size=(30, 3)), 199, rng)
    assert 0.8 < ratio0 < 1.2 and p0 > 0.01


def test_knn_purity_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    lab = rng.integers(0, 3, 300)
    assert knn_purity(X, lab) == pytest.approx(1 / 3, abs=0.06)
