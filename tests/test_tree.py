"""Ground-truth tests for the RefTree reference phylogeny structure."""

import dendropy
import numpy as np
import pytest

from phylogeo.data.tree import RefTree

# ((A,B),(C,(D,E))) with a unifurcation chain above (A,B) like Open Tree emits.
NWK = "((((A,B)x)y,(C,(D,E)z)w)r);"


def test_unifurcations_suppressed_and_leaf_order():
    t = RefTree(NWK, ["E", "D", "C", "B", "A"])
    assert t.n == 5
    # Binary tree with 5 leaves has 4 internal nodes after suppression.
    assert t.N == 9
    assert t.leaf_labels == ["E", "D", "C", "B", "A"]


def test_lca_and_path_length_match_dendropy():
    order = ["A", "B", "C", "D", "E"]
    t = RefTree(NWK, order)
    D = t.path_length()
    # Hand-computed edge counts on ((A,B),(C,(D,E))).
    assert D[0, 1] == 2  # A-B
    assert D[3, 4] == 2  # D-E
    assert D[2, 3] == 3  # C-D
    assert D[0, 2] == 4  # A-C
    assert D[0, 4] == 5  # A-E
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0)
    # Cross-check with DendroPy's unweighted patristic distances.
    dt = dendropy.Tree.get(data=NWK, schema="newick")
    dt.suppress_unifurcations()
    for e in dt.preorder_edge_iter():
        e.length = 1.0
    pdm = dt.phylogenetic_distance_matrix()
    tax = {x.label: x for x in dt.taxon_namespace}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i != j:
                assert D[i, j] == pytest.approx(pdm.distance(tax[a], tax[b]))


def test_ultrametric_from_ages():
    order = ["A", "B", "C", "D", "E"]
    t = RefTree(NWK, order)
    age = np.zeros(t.N)
    for v in t.internal_nodes:
        m = set(t.leaf_labels[i] for i in t.members[v])
        age[v] = {frozenset("AB"): 10, frozenset("DE"): 5, frozenset("CDE"): 20,
                  frozenset("ABCDE"): 50}[frozenset(m)]
    U = t.ultrametric(age)
    assert U[0, 1] == 20 and U[3, 4] == 10 and U[2, 4] == 40 and U[0, 4] == 100


def test_polytomy_and_subtree():
    t = RefTree("((A,B,C),D);", ["A", "B", "C", "D"])
    assert t.N == 6  # 4 leaves + polytomy node + root
    assert len(t.children[t.lca[0, 1]]) == 3
    s = t.subtree([0, 1, 3])  # drop C -> ((A,B),D)
    assert s.n == 3 and s.path_length()[0, 1] == 2


def test_round_trip_newick():
    t = RefTree(NWK, ["A", "B", "C", "D", "E"])
    t2 = RefTree(t.to_newick(), ["A", "B", "C", "D", "E"])
    assert np.array_equal(t.path_length(), t2.path_length())


def test_bad_leaf_order_raises():
    with pytest.raises(ValueError):
        RefTree(NWK, ["A", "B", "C"])
