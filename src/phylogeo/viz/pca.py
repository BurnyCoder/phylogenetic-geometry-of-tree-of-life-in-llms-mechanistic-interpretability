"""Stage 1a: top-3 PCA views of species representations and the H1a "is it a tree?" checks.

Global context: the neural-geometry practice is to first *look* at the top PCA
directions (days of the week form a circle there; Engels et al. 2024,
https://arxiv.org/abs/2405.14860). This module computes the top-3 PCs per layer,
overlays the reference tree in PC space (internal nodes at the centroid of their
descendant leaves), and computes the preregistered checks:
  C1 stratum kNN purity / silhouette, C2 within-stratum order purity,
  C3 tree-overlay length vs shuffled leaf labels, C4 PC correlations with
  strata and confounds. Figures: interactive Plotly 3D with a layer slider,
  matplotlib 2D panels (colored and uncolored for the anti-bias step).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from phylogeo.data.tree import RefTree

# Composite encoding for 13 strata in scatter plots (dataviz rule: only 3 categorical
# hues stay distinguishable across all pairs in a scatter; validated with the skill's
# validate_palette.js --pairs all: CVD dE >= 9.2, normal-vision dE >= 24.0).
# Color = super-group, marker = stratum within the group; "other" is neutral gray.
VERT, ARTH, INV, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
# stratum -> (color, matplotlib marker, plotly 3D symbol)
STRATUM_STYLE = {
    "mammals": (VERT, "o", "circle"), "birds": (VERT, "^", "diamond"), "reptiles": (VERT, "s", "square"),
    "amphibians": (VERT, "D", "cross"), "ray-finned fish": (VERT, "v", "x"),
    "cartilaginous fish": (VERT, "P", "circle-open"),
    "insects": (ARTH, "o", "circle"), "arachnids": (ARTH, "^", "diamond"), "crustaceans": (ARTH, "s", "square"),
    "molluscs": (INV, "o", "circle"), "cnidarians": (INV, "^", "diamond"), "echinoderms": (INV, "s", "square"),
    "other": (OTHER, "x", "x"),
}
STRATUM_COLORS = {k: v[0] for k, v in STRATUM_STYLE.items()}


def pca_top3(V: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on [n, d] (centered internally) and return (coords [n, 3], explained ratio [3], model)."""
    p = PCA(n_components=3, random_state=seed).fit(V)
    return p.transform(V), p.explained_variance_ratio_, p


def overlay_positions(tree: RefTree, coords: np.ndarray) -> np.ndarray:
    """[N, k] positions: leaves at their coords, internal nodes at the mean of descendant leaves."""
    pos = np.zeros((tree.N, coords.shape[1]))
    pos[: tree.n] = coords
    for v in tree.internal_nodes:
        pos[v] = coords[tree.members[v]].mean(0)
    return pos


def overlay_length(tree: RefTree, coords: np.ndarray) -> float:
    """Sum of Euclidean edge lengths of the tree drawn with :func:`overlay_positions`."""
    pos = overlay_positions(tree, coords)
    child = np.arange(tree.N) != tree.root
    return float(np.linalg.norm(pos[child] - pos[tree.parent[child]], axis=1).sum())


def overlay_length_test(tree: RefTree, coords: np.ndarray, n_perm: int, rng: np.random.Generator,
                        groups: np.ndarray | None = None) -> tuple[float, float, float]:
    """C3: observed overlay length vs shuffled leaf labels (optionally within groups).

    Returns (observed, mean null, one-sided p for "shorter than null").
    """
    obs = overlay_length(tree, coords)
    g = np.zeros(tree.n, int) if groups is None else np.asarray(groups)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = np.arange(tree.n)
        for gv in np.unique(g):
            idx = np.where(g == gv)[0]
            p[idx] = rng.permutation(idx)
        null[i] = overlay_length(tree, coords[p])
    return obs, float(null.mean()), float((1 + np.sum(null <= obs)) / (n_perm + 1))


def knn_purity(coords: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Mean fraction of each point's k nearest neighbors (Euclidean) sharing its label."""
    lab = np.asarray(labels)
    D = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    return float((lab[nn] == lab[:, None]).mean())


def purity_test(coords: np.ndarray, labels: np.ndarray, n_perm: int, rng: np.random.Generator,
                k: int = 5) -> tuple[float, float, float]:
    """kNN purity vs label permutation: (observed, mean null, one-sided p for "higher")."""
    obs = knn_purity(coords, labels, k)
    lab = np.asarray(labels)
    null = np.array([knn_purity(coords, rng.permutation(lab), k) for _ in range(n_perm)])
    return obs, float(null.mean()), float((1 + np.sum(null >= obs)) / (n_perm + 1))


def h1a_checks(V: np.ndarray, tree: RefTree, strata: np.ndarray, orders: np.ndarray,
               confounds: dict[str, np.ndarray], n_perm: int, rng: np.random.Generator,
               zoom_strata=("mammals", "birds", "insects")) -> dict:
    """All preregistered stage-1a checks for one layer's species vectors V [n, d]."""
    coords, evr, _ = pca_top3(V)
    strata = np.asarray(strata)
    orders = np.asarray(orders)
    out: dict = {"explained_variance_ratio": evr.tolist()}
    c1, c1null, c1p = purity_test(coords, strata, n_perm, rng)
    out["C1_purity"], out["C1_purity_null"], out["C1_p"] = c1, c1null, c1p
    out["C1_silhouette"] = float(silhouette_score(coords, strata))
    c2 = {}
    for s in zoom_strata:
        m = strata == s
        if m.sum() < 10:
            continue
        sub = PCA(n_components=3, random_state=0).fit_transform(V[m])
        o, on, op = purity_test(sub, orders[m], n_perm, rng)
        c2[s] = {"purity": o, "null": on, "p": op, "n": int(m.sum()), "n_orders": int(len(set(orders[m])))}
    out["C2_within_stratum_order_purity"] = c2
    c3, c3null, c3p = overlay_length_test(tree, coords, n_perm, rng)
    out["C3_overlay_length"], out["C3_null_mean"], out["C3_p"] = c3, c3null, c3p
    # C3b (exploratory, added after the pilot): shuffle leaves only *within* strata, so
    # class clustering alone cannot shorten the overlay -> tests branching beyond classes.
    _, c3bnull, c3bp = overlay_length_test(tree, coords, n_perm, rng, groups=strata)
    out["C3b_within_stratum_null_mean"], out["C3b_p"] = c3bnull, c3bp
    c4 = {}
    for pc in range(3):
        row = {}
        for s in np.unique(strata):
            row[f"stratum:{s}"] = float(spearmanr(coords[:, pc], strata == s).statistic)
        for name, vals in confounds.items():
            row[name] = float(spearmanr(coords[:, pc], vals).statistic)
        c4[f"PC{pc + 1}"] = row
    out["C4_pc_correlations"] = c4
    return out


def classify_h1a(checks: dict, alpha: float = 0.05) -> str:
    """Preregistered reading of one layer's checks (shape judgment is added by eye)."""
    c1 = checks["C1_p"] < alpha
    # C2 must hold in every zoom stratum (mammals, birds, insects): stricter reading of
    # "C2 beats its null" (review finding: "any stratum" was looser than registered).
    c2 = all(v["p"] < alpha for v in checks["C2_within_stratum_order_purity"].values())
    c3 = checks["C3_p"] < alpha
    if not c1:
        return "no structure"
    if c1 and c2 and c3:
        return "tree-consistent (checks 1-3 pass)"
    return "clusters, not a tree (C1 passes; C2/C3 fail)" if not (c2 or c3) else "partial (C1 + one of C2/C3)"
