"""Compact derived data for the interactive figures (phase ``viz_data``; CPU only).

Global context: the site and README figures must be regenerated from ``results/`` alone
(the activations live on D: and on Hugging Face, not in git). This phase reads the stored
activations once and writes three small files to ``results/<run>/viz/``:

- ``distances.npz``: species x species distance *percentiles* (uint8, 0 = closest pair) for the
  reference tree, the model, its random-init twin and the class-only oracle, rows and columns in
  depth-first leaf order of the reference tree, so that clades are contiguous blocks.
- ``pca2d.json``: 2-D PCA coordinates of the expected shape (Brownian motion on the reference
  tree, the same synthetic positive control as stage 1a) and of the observed representations.
- ``neighbors.json``: the 10 nearest species of every species under the tree, the model and the
  untrained controls, each labelled with its taxonomic relation to the query.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import rankdata
from sklearn.decomposition import PCA

log = logging.getLogger("phylogeo.viz_data")

# Relation of a neighbour to the query, from closest to farthest (first shared Linnaean rank).
LEVELS = ["genus", "family", "order", "class", "phylum"]
LEVEL_NAMES = ["same genus", "same family", "same order", "same class", "same phylum", "different phylum"]


def dfs_leaf_order(tree) -> list[int]:
    """Leaf indices in depth-first order, so every clade is a contiguous block.
    (``RefTree.preorder`` is breadth-first, which would interleave clades.)"""
    out, stack = [], [tree.root]
    while stack:
        v = stack.pop()
        if v < tree.n:
            out.append(v)
        else:
            stack.extend(reversed(tree.children[v]))  # reversed: children are visited in stored order
    return out


def percentile_matrix(D: np.ndarray) -> np.ndarray:
    """Off-diagonal entries -> percentile rank 0..100 (ties averaged), uint8, symmetric, diagonal 0.
    Percentiles put matrices with different units (edges, cosine) on one colour scale."""
    n = len(D)
    iu = np.triu_indices(n, 1)
    r = rankdata(D[iu], method="average")  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
    P = np.zeros((n, n))
    P[iu] = 100 * (r - 1) / max(len(r) - 1, 1)
    return np.rint(P + P.T).astype(np.uint8)


def relation_level(tax: dict[str, np.ndarray], i: int, j: int) -> int:
    """Index into LEVEL_NAMES: the lowest Linnaean rank shared by species i and j (empty = unknown)."""
    for k, lv in enumerate(LEVELS):
        a, b = tax[lv][i], tax[lv][j]
        if a and a == b:
            return k
    return len(LEVELS)


def nearest(D: np.ndarray, k: int) -> np.ndarray:
    """[n, k] indices of the k nearest other species (stable sort: ties keep table order)."""
    D = D.astype(float).copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def build_viz_data(main: str, controls: list[str], layers: list[int], out: Path, seed: int = 0, k: int = 10) -> None:
    """Write distances.npz, pca2d.json and neighbors.json for ``main`` (see module docstring)."""
    from phylogeo.baselines import class_oracle
    from phylogeo.geometry import cosine_distance, species_vectors
    from phylogeo.phases import act_dir, load_species
    from phylogeo.extract import load_activations
    from phylogeo.simulate import brownian_tree

    out.mkdir(parents=True, exist_ok=True)
    df, tree = load_species()
    names = df["common_name"].tolist()
    strata = df["stratum"].to_numpy()
    tax = {lv: df[lv].to_numpy() for lv in LEVELS}
    L = layers[0]  # the confirmatory (modal) layer comes first
    V = {r: species_vectors(load_activations(act_dir(r, "species"), "last")) for r in [main, *controls]}
    D = {r: cosine_distance(v[L]) for r, v in V.items()}
    P = tree.path_length()
    order = np.array(dfs_leaf_order(tree))
    ix = np.ix_(order, order)

    # 1. Distance percentiles in tree order. The class oracle is binary, so it is stored as 0 / 100 directly.
    mats = {"tree": percentile_matrix(P)[ix], "model": percentile_matrix(D[main])[ix],
            "untrained": percentile_matrix(D[controls[0]])[ix],
            "class_only": (100 * class_oracle(strata)).astype(np.uint8)[ix]}
    np.savez_compressed(out / "distances.npz", order=order, **mats)

    # 2. 2-D PCA: expected (Brownian motion on the tree, same generator as the stage-1a control) vs observed.
    def pca2(X):
        p = PCA(n_components=2, random_state=seed).fit(X - X.mean(0))
        return {"xy": np.round(p.transform(X - X.mean(0)), 4).tolist(), "evr": p.explained_variance_ratio_.tolist()}

    Xb = brownian_tree(tree, V[main].shape[-1], np.random.default_rng([seed, 11]))
    pcs = {"expected": pca2(Xb)}
    for lay in layers:
        pcs[f"{main}@{lay}"] = pca2(V[main][lay])
    pcs[f"{controls[0]}@{L}"] = pca2(V[controls[0]][L])
    (out / "pca2d.json").write_text(json.dumps({"names": names, "strata": strata.tolist(), "panels": pcs}),
                                    encoding="utf-8")

    # 3. Nearest neighbours with their relation to the query.
    reps = {"tree": P, main: D[main], **{c: D[c] for c in controls}}
    nn = {r: nearest(M, k) for r, M in reps.items()}
    lv = {r: [[relation_level(tax, i, int(j)) for j in row] for i, row in enumerate(idx)] for r, idx in nn.items()}
    (out / "neighbors.json").write_text(json.dumps({
        "layer": L, "k": k, "names": names, "strata": strata.tolist(), "level_names": LEVEL_NAMES,
        "taxonomy": {c: df[c].tolist() for c in ["phylum", "class", "order", "family", "genus", "sci_name"]},
        "neighbors": {r: idx.tolist() for r, idx in nn.items()}, "levels": lv}), encoding="utf-8")
    # 4. Top-3 PCA at every layer, aligned layer to layer, for the animated 3-D view (stage-1a look).
    p3 = {r: aligned_pca3_layers(V[r], seed) for r in [main, controls[0]]}
    eb, evb = pca3(Xb, seed)
    R, _ = orthogonal_procrustes(eb, np.array(p3[main]["coords"][L]))  # expected shape, rotated onto the model's view
    p3["expected"] = {"coords": [np.round(eb @ R, 3).tolist()], "evr": [evb]}
    (out / "pca3d.json").write_text(json.dumps({
        "names": names, "strata": strata.tolist(), "modal_layer": L, "reps": p3,
        "tree": {"n": int(tree.n), "parent": tree.parent.tolist(), "postorder": list(reversed(tree.preorder))}}),
        encoding="utf-8")
    log.info("viz data written to %s (layers %s, controls %s)", out, layers, controls)


def pca3(X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Top-3 PCA coordinates scaled so the *median* species radius is 1, and their explained-variance ratios.
    The median is used instead of the RMS radius because in the untrained network a few outlier names dominate
    the RMS, which would shrink every other species to a dot; with one robust rule, both models fill the same box."""
    p = PCA(n_components=3, random_state=seed).fit(X - X.mean(0))
    C = p.transform(X - X.mean(0))
    return C / np.median(np.linalg.norm(C, axis=1)), p.explained_variance_ratio_.tolist()


def aligned_pca3_layers(V: np.ndarray, seed: int = 0) -> dict:
    """[L, n, d] species vectors -> per-layer top-3 PCA coordinates, each layer rotated (orthogonal Procrustes,
    reflections allowed) onto the previous one. PCA axes are defined only up to sign and order, so without this
    the animation would flip and jump between layers. The rotation is display-only: within-layer distances and
    explained variance are unchanged. scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html"""
    coords, evr, prev = [], [], None
    for X in V:
        C, e = pca3(X, seed)
        if prev is not None:
            R, _ = orthogonal_procrustes(C, prev)  # R minimizes ||C R - prev||_F over orthogonal R
            C = C @ R
        coords.append(np.round(C, 3).tolist())
        evr.append([round(x, 4) for x in e])
        prev = C
    return {"coords": coords, "evr": evr}
