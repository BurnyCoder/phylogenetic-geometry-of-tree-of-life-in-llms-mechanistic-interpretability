"""Reference phylogeny as an indexed rooted tree, plus TimeTree dating.

Global context: :class:`RefTree` is the ground truth used by every tree metric.
It is built from the Open Tree induced-subtree Newick (unary nodes suppressed so
edge counts are meaningful), with leaves ordered like ``data/species.csv``. It
pre-computes the lowest-common-ancestor (LCA) matrix, from which come
(i) triplet resolution (stats/triplets.py), (ii) path-length distances (Mantel,
S1), and (iii) ultrametric divergence-time distances once internal nodes are
dated with TimeTree 5 (S5; https://timetree.temple.edu/api; data cached
locally only because TimeTree does not permit redistribution).
"""

from __future__ import annotations

import logging

import dendropy
import numpy as np

from phylogeo.webcache import request_json

log = logging.getLogger("phylogeo.tree")


class RefTree:
    """Rooted tree with leaves 0..n-1 (in a given order) and internal nodes n..N-1.

    Attributes
    ----------
    leaf_labels : list[str]   labels of leaves in index order (e.g. "ott770315")
    parent : np.ndarray       parent node index of every node (-1 for the root)
    children : list[list[int]] child node indices of every node ([] for leaves)
    depth : np.ndarray        number of edges from the root to each node
    members : list[np.ndarray] leaf indices below each node (a leaf contains itself)
    lca : np.ndarray          [n, n] node index of LCA(i, j) (lca[i, i] = i)
    node_labels : list[str]   Newick labels of all nodes ("" if none)
    """

    def __init__(self, newick: str, leaf_order: list[str] | None = None):
        # Parse; keep underscores so labels like "ott770315" stay intact.
        t = dendropy.Tree.get(data=newick, schema="newick", preserve_underscores=True,
                              suppress_internal_node_taxa=True, suppress_leaf_node_taxa=False)
        # Remove unary (one-child) nodes: Open Tree emits long chains of them, and
        # they would inflate path lengths without carrying branching information.
        t.suppress_unifurcations()
        leaves = t.leaf_nodes()
        labels = [nd.taxon.label for nd in leaves]
        if leaf_order is None:
            leaf_order = labels
        if sorted(leaf_order) != sorted(labels):
            missing = set(leaf_order) ^ set(labels)
            raise ValueError(f"leaf_order does not match tree leaves; diff={sorted(missing)[:10]}")
        pos = {lab: i for i, lab in enumerate(leaf_order)}
        n = len(leaf_order)
        # Index nodes: leaves get their position in leaf_order, internal nodes are
        # numbered n, n+1, ... in postorder (children before parents).
        idx: dict[int, int] = {}
        for nd in leaves:
            idx[id(nd)] = pos[nd.taxon.label]
        nxt = n
        internal = []
        for nd in t.postorder_node_iter():
            if nd.is_leaf():
                continue
            idx[id(nd)] = nxt
            internal.append(nd)
            nxt += 1
        N = nxt
        self.n = n
        self.N = N
        self.leaf_labels = list(leaf_order)
        self.parent = np.full(N, -1, dtype=int)
        self.children: list[list[int]] = [[] for _ in range(N)]
        self.node_labels = [""] * N
        for nd in t.preorder_node_iter():
            i = idx[id(nd)]
            self.node_labels[i] = nd.taxon.label if nd.is_leaf() else (nd.label or "")
            for c in nd.child_node_iter():
                self.children[i].append(idx[id(c)])
                self.parent[idx[id(c)]] = i
        self.root = int(np.where(self.parent == -1)[0][0])
        # Depth (edges from root) by preorder traversal.
        self.depth = np.zeros(N, dtype=int)
        order = [self.root]
        for v in order:
            for c in self.children[v]:
                self.depth[c] = self.depth[v] + 1
                order.append(c)
        self.preorder = order
        # Leaf members of every node (postorder accumulation).
        self.members: list[np.ndarray] = [np.array([], dtype=int)] * N
        for v in reversed(order):
            if not self.children[v]:
                self.members[v] = np.array([v], dtype=int)
            else:
                self.members[v] = np.concatenate([self.members[c] for c in self.children[v]])
        # LCA matrix: for each internal node v, pairs of leaves in *different*
        # children of v have LCA v. Every leaf pair is assigned exactly once.
        self.lca = np.empty((n, n), dtype=int)
        np.fill_diagonal(self.lca, np.arange(n))
        for v in range(n, N):
            ch = self.children[v]
            for a in range(len(ch)):
                for b in range(a + 1, len(ch)):
                    ia, ib = self.members[ch[a]], self.members[ch[b]]
                    self.lca[np.ix_(ia, ib)] = v
                    self.lca[np.ix_(ib, ia)] = v

    @property
    def internal_nodes(self) -> range:
        """Indices of internal nodes (n .. N-1)."""
        return range(self.n, self.N)

    def path_length(self) -> np.ndarray:
        """[n, n] number of edges on the path between leaves (topological distance)."""
        d = self.depth
        return (d[:, None][: self.n] + d[None, : self.n] - 2 * d[self.lca]).astype(float)

    def lca_depth_distance(self) -> np.ndarray:
        """[n, n] topological ultrametric: (max depth - depth of LCA), 0 on the diagonal.

        Unlike edge-count path length, this distance is consistent with every rooted
        triplet of the tree (the tree-closer pair always has the smaller distance),
        so it is the natural "oracle" for triplet accuracy and rank-like structure.
        """
        D = (self.depth.max() - self.depth[self.lca]).astype(float)
        np.fill_diagonal(D, 0.0)
        return D

    def ultrametric(self, node_age: np.ndarray) -> np.ndarray:
        """[n, n] patristic distance 2 * age(LCA) for a dated tree (leaves have age 0)."""
        D = 2.0 * node_age[self.lca]
        np.fill_diagonal(D, 0.0)
        return D

    def to_newick(self) -> str:
        """Write the (unifurcation-suppressed) tree back to Newick with leaf labels."""

        def rec(v: int) -> str:
            if not self.children[v]:
                return self.leaf_labels[v]
            return "(" + ",".join(rec(c) for c in self.children[v]) + ")"

        return rec(self.root) + ";"

    def subtree(self, keep: list[int]) -> "RefTree":
        """Induced subtree on a subset of leaf indices (order preserved)."""
        keep_set = set(keep)

        def rec(v: int) -> str | None:
            if not self.children[v]:
                return self.leaf_labels[v] if v in keep_set else None
            parts = [s for s in (rec(c) for c in self.children[v]) if s is not None]
            if not parts:
                return None
            return parts[0] if len(parts) == 1 else "(" + ",".join(parts) + ")"

        return RefTree(rec(self.root) + ";", [self.leaf_labels[i] for i in keep])


def timetree_mrca_age(ncbi_ids: list[int]) -> float | None:
    """Divergence time (Mya) of the MRCA of NCBI taxa from TimeTree 5 (cached).

    Local: GET /api/mrca/id/<id+id+...>/json -> ``precomputed_age``. Returns None if
    TimeTree has no estimate (missing ids or no studies).
    """
    ids = "+".join(str(int(i)) for i in ncbi_ids)
    try:
        d = request_json("GET", f"https://timetree.temple.edu/api/mrca/id/{ids}/json",
                         namespace="timetree_mrca", accept_error_json=True, retries=3)
    except RuntimeError:
        return None
    age = d.get("precomputed_age") if isinstance(d, dict) else None
    if age is None or (d.get("missing_ids") and len(d.get("found_ids", [])) < 2):
        return None
    return float(age)


def date_tree(tree: RefTree, ncbi_by_leaf: list[int], tries: int = 8) -> np.ndarray:
    """Date every internal node with TimeTree; return node ages (Mya), leaves = 0.

    Local: for internal node v, pick one representative leaf from each of two
    different children (up to ``tries`` different pairs) and query the MRCA age.
    Undated nodes inherit the age of their oldest dated child (or 0). Ages are
    then made monotone (a parent is never younger than any child).
    Global: produces the dated ultrametric tree for secondary hypothesis S5.
    """
    age = np.zeros(tree.N)  # leaves = 0 Mya
    dated = np.zeros(tree.N, dtype=bool)
    dated[: tree.n] = True
    rng = np.random.default_rng(0)
    for v in tree.internal_nodes:
        ch = tree.children[v]
        for _ in range(tries):
            a, b = rng.choice(len(ch), size=2, replace=False)
            la = int(rng.choice(tree.members[ch[a]]))
            lb = int(rng.choice(tree.members[ch[b]]))
            t = timetree_mrca_age([ncbi_by_leaf[la], ncbi_by_leaf[lb]])
            if t is not None:
                age[v], dated[v] = t, True
                break
    # Postorder pass (internal nodes are numbered in postorder): a dated node is
    # never younger than any dated descendant (TimeTree estimates for different
    # pairs can be slightly inconsistent). Undated nodes stay NaN: they are NOT
    # given a child's age (that produced impossible 0-Mya splits; review finding 4).
    max_desc = np.where(dated, age, -np.inf)
    for v in tree.internal_nodes:
        m = max(max_desc[c] for c in tree.children[v])
        if dated[v]:
            age[v] = max(age[v], m)
            max_desc[v] = age[v]
        else:
            max_desc[v] = m
    age[~dated] = np.nan
    log.info("TimeTree dated %d / %d internal nodes (others NaN)", int(dated[tree.n:].sum()), tree.N - tree.n)
    return age
