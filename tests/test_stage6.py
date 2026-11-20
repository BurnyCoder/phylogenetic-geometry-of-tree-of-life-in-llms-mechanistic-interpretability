"""Stage-6 helpers: spread summaries and margin over the best control."""

import numpy as np

from phylogeo.stage6 import _per_ref, summarize


def test_summarize_matches_numpy():
    x = [0.4, 0.5, 0.45, 0.47]
    s = summarize(x)
    assert np.isclose(s["mean"], np.mean(x)) and np.isclose(s["sd"], np.std(x, ddof=1))
    assert s["min"] == 0.4 and s["max"] == 0.5


def test_margin_uses_best_control_per_repetition():
    vals = {"model": [0.5, 0.45], "c1": [0.40, 0.44], "c2": [0.42, 0.30]}
    out = _per_ref(vals, "model")
    assert np.allclose(out["margin_vs_best_control"]["values"], [0.08, 0.01])
    assert out["best_control_by_rep"] == ["c2", "c1"]


def test_subtree_preserves_requested_leaf_order(tree30):
    keep = [5, 1, 3, 8]
    sub = tree30.subtree(keep)
    assert sub.leaf_labels == [tree30.leaf_labels[i] for i in keep]


def test_nearest_and_same_label_rate():
    from phylogeo.examples import nearest, same_label_rate

    D = np.array([[0, 1, 5, 6], [1, 0, 5, 6], [5, 5, 0, 1], [6, 6, 1, 0]], float)
    assert nearest(D, 0, 2) == [1, 2]
    assert same_label_rate(D, np.array(["a", "a", "b", "b"], dtype=object), k=1) == 1.0
    assert same_label_rate(D, np.array(["a", "b", "a", "b"], dtype=object), k=1) == 0.0
