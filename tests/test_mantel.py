"""Tests for Mantel statistics against scikit-bio and synthetic ground truth."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from skbio import DistanceMatrix
from skbio.stats.distance import mantel as skbio_mantel

from phylogeo.stats.mantel import kendall_tau_b, mantel_maxT, mantel_r, partial_mantel


def _dm(rng, n=25, d=5):
    X = rng.normal(size=(n, d))
    return X, cdist(X, X)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_mantel_r_matches_skbio(rng, method):
    X, D1 = _dm(rng)
    D2 = cdist(X + 0.5 * rng.normal(size=X.shape), X + 0.5 * rng.normal(size=X.shape))
    D2 = (D2 + D2.T) / 2
    np.fill_diagonal(D2, 0)
    r_sk, _, _ = skbio_mantel(DistanceMatrix(D1), DistanceMatrix(D2), method=method, permutations=0)
    assert mantel_r(D1, D2, method) == pytest.approx(r_sk, abs=1e-10)


def test_maxT_signal_and_null(rng):
    X, D = _dm(rng, n=30)
    _, Y = _dm(rng, n=30)
    obs, p, _ = mantel_maxT(np.stack([Y, D, Y]), D, n_perm=199, rng=rng)
    assert obs.argmax() == 1 and obs[1] == pytest.approx(1.0) and p < 0.01
    ps = []
    for s in range(30):
        r = np.random.default_rng(s)
        _, A = _dm(r, n=20)
        _, B = _dm(r, n=20)
        ps.append(mantel_maxT(np.stack([A]), B, n_perm=99, rng=r)[1])
    assert np.mean(np.array(ps) < 0.05) <= 0.2  # not anti-conservative under the null
    assert 0.25 < np.mean(ps) < 0.75  # p-values spread around 0.5


def test_partial_mantel_mediation(rng):
    n, d = 30, 4
    Z = rng.normal(size=(n, d))
    # X and Y both derive from Z only -> partial r(X, Y | Z) ~ 0.
    X = Z + 0.3 * rng.normal(size=(n, d))
    Y = Z + 0.3 * rng.normal(size=(n, d))
    r, p = partial_mantel(cdist(X, X), cdist(Y, Y), cdist(Z, Z), n_perm=199, rng=rng)
    assert abs(r) < 0.25
    # Direct effect: Y shares W with X beyond Z -> partial r > 0 and significant.
    W = rng.normal(size=(n, d))
    X2, Y2 = np.hstack([Z, W]), np.hstack([Z, W]) + 0.3 * rng.normal(size=(n, 2 * d))
    r2, p2 = partial_mantel(cdist(X2, X2), cdist(Y2, Y2), cdist(Z, Z), n_perm=199, rng=rng)
    assert r2 > 0.3 and p2 < 0.01


def test_kendall_tau_b_identity(rng):
    _, D = _dm(rng)
    assert kendall_tau_b(D, D) == pytest.approx(1.0)
