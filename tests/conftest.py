"""Shared fixtures: random rooted trees and tree-structured point clouds."""

import numpy as np
import pytest

from phylogeo.data.tree import RefTree


def random_binary_newick(labels: list[str], rng: np.random.Generator) -> str:
    """Random rooted binary tree over ``labels`` by recursive random bipartition."""

    def rec(ls: list[str]) -> str:
        if len(ls) == 1:
            return ls[0]
        k = int(rng.integers(1, len(ls)))
        perm = list(rng.permutation(ls))
        return "(" + rec(perm[:k]) + "," + rec(perm[k:]) + ")"

    return rec(labels) + ";"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def tree30(rng):
    labels = [f"L{i}" for i in range(30)]
    return RefTree(random_binary_newick(labels, rng), labels)


def brownian_on_tree(tree: RefTree, dim: int, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    """Brownian motion along the tree: each edge adds an N(0, I) step; leaves get
    the sum along their root path (+ optional isotropic noise)."""
    X = np.zeros((tree.N, dim))
    for v in tree.preorder:
        for c in tree.children[v]:
            X[c] = X[v] + rng.normal(size=dim)
    return X[: tree.n] + noise * rng.normal(size=(tree.n, dim))
