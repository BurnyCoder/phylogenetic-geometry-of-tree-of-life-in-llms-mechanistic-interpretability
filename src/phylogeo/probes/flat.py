"""Goodfire-style "flat subspace" metric probe with family-holdout cross-validation.

Global context (stage 2): following Goodfire's Evo 2 analysis
(https://www.goodfire.com/research/phylogeny-manifold), a linear map projects
species vectors to k dimensions, z = W (x - b), and the angular distance between
projected vectors, scaled by a learned beta, is trained to match tree distance:
    d_hat(i, j) = beta * arccos(cos(z_i, z_j)),
    loss = mean((d_hat - d_tree)^2) + alpha * ||pinv(W) z + b - x||^2.
Generalization is measured on held-out *families* (no species of a test family
is seen in training), and selectivity against a shuffled-tree control probe
(Hewitt & Liang 2019, https://aclanthology.org/D19-1275/). Inputs are PCA-100
features fit on the training fold only. Training metrics stream to trackio.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

log = logging.getLogger("phylogeo.probes")


class FlatProbe(torch.nn.Module):
    """z = W (x - b); d_hat = beta * arccos(cos(z_i, z_j)); reconstruction via pinv(W)."""

    def __init__(self, p: int, k: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(k, p, generator=g) / np.sqrt(p))
        self.b = torch.nn.Parameter(torch.zeros(p))
        self.log_beta = torch.nn.Parameter(torch.zeros(()))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b) @ self.W.T

    def dist(self, z: torch.Tensor) -> torch.Tensor:
        u = torch.nn.functional.normalize(z, dim=-1)
        cos = (u @ u.T).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.exp(self.log_beta) * torch.arccos(cos)

    def recon(self, z: torch.Tensor) -> torch.Tensor:
        return z @ torch.linalg.pinv(self.W).T + self.b


def train_probe(X: np.ndarray, D: np.ndarray, k: int = 10, steps: int = 2000, lr: float = 1e-3,
                alpha: float = 0.01, seed: int = 0, tracker=None, tag: str = "") -> FlatProbe:
    """Fit a FlatProbe on all pairs of the given species (full batch)."""
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    d = torch.tensor(D, dtype=torch.float32)
    iu = torch.triu_indices(len(X), len(X), 1)
    m = FlatProbe(X.shape[1], k, seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for step in range(steps):
        z = m.project(x)
        dh = m.dist(z)
        l_d = ((dh[iu[0], iu[1]] - d[iu[0], iu[1]]) ** 2).mean()
        l_r = ((m.recon(z) - x) ** 2).mean()
        loss = l_d + alpha * l_r
        opt.zero_grad()
        loss.backward()
        opt.step()
        if tracker is not None and (step % 50 == 0 or step == steps - 1):
            tracker.log({f"{tag}/loss": float(loss), f"{tag}/dist_mse": float(l_d), f"{tag}/recon": float(l_r)},
                        step=step)
    return m


def probe_distances(m: FlatProbe, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return m.dist(m.project(torch.tensor(X, dtype=torch.float32))).numpy()


def family_folds(families: np.ndarray, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Assign whole families to folds; returns a boolean test mask per fold."""
    uf = rng.permutation(np.unique(families))
    fold_of = {f: i % n_folds for i, f in enumerate(uf)}
    fid = np.array([fold_of[f] for f in families])
    return [fid == i for i in range(n_folds)]


def cv_probe(V: np.ndarray, D_tree: np.ndarray, families: np.ndarray, k: int = 10, n_folds: int = 5,
             n_pca: int = 100, steps: int = 2000, seed: int = 0, tracker=None, tag: str = "probe") -> dict:
    """Family-holdout CV of the flat probe; returns test-test / train-test correlations.

    Local: per fold, PCA is fit on training species only; the target is the tree
    distance rescaled to [0, pi]; correlations are computed on held-out pairs.
    """
    rng = np.random.default_rng(seed)
    scale = np.pi / D_tree.max()
    res = {"test_test_pearson": [], "test_test_spearman": [], "train_test_pearson": [], "train_train_pearson": []}
    for fi, te in enumerate(family_folds(families, n_folds, rng)):
        tr = ~te
        pca = PCA(n_components=min(n_pca, tr.sum() - 1), random_state=seed).fit(V[tr])
        Z = pca.transform(V)
        Z = Z / Z[tr].std()
        m = train_probe(Z[tr], D_tree[np.ix_(tr, tr)] * scale, k=k, steps=steps, seed=seed + fi,
                        tracker=tracker, tag=f"{tag}/fold{fi}")
        Dh = probe_distances(m, Z)
        tt = np.triu_indices(te.sum(), 1)
        a, b = Dh[np.ix_(te, te)][tt], D_tree[np.ix_(te, te)][tt]
        res["test_test_pearson"].append(float(pearsonr(a, b).statistic))
        res["test_test_spearman"].append(float(spearmanr(a, b).statistic))
        res["train_test_pearson"].append(float(pearsonr(Dh[np.ix_(tr, te)].ravel(), D_tree[np.ix_(tr, te)].ravel()).statistic))
        trr = np.triu_indices(tr.sum(), 1)
        res["train_train_pearson"].append(float(pearsonr(Dh[np.ix_(tr, tr)][trr], D_tree[np.ix_(tr, tr)][trr]).statistic))
    return {k_: (float(np.mean(v)), float(np.std(v))) for k_, v in res.items()}
