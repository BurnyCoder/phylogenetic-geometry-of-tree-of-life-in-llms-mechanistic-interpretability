"""Model-free regression checks for the estimands audited in November 2026."""

import numpy as np

from phylogeo.verification.statistics import pair_scores, transition_bounds, holm_family


def test_registered_ties_and_historical_minimum_sharing():
    # Columns cover a strict win, a strict loss, both kinds of two-way tie and a three-way tie.
    sister = np.array([0, 3, 0, 3, 1])
    first = np.array([1, 0, 0, 0, 1])
    second = np.array([2, 1, 2, 0, 1])
    np.testing.assert_allclose(pair_scores(sister, first, second), [1, 0, 1/3, 1/3, 1/3])
    np.testing.assert_allclose(pair_scores(sister, first, second, historical=True), [1, 0, .5, 0, 1/3])


def test_transitions_from_aggregate_rates():
    assert transition_bounds(200, 4, 11) == [7, 11]
    assert transition_bounds(10, 9, 8) == [0, 1]


def test_unevaluable_slot_counts_in_holm_family():
    result = holm_family({"a": .01, "b": None, "c": .02, "d": .7})
    assert result == {"a": .04, "b": None, "c": .06, "d": 1.0}


def test_dating_midrank_optimization_preserves_ties():
    from scipy.stats import rankdata
    from phylogeo.verification.dating import centered_ranks

    keys = np.random.default_rng(42).integers(0, 60, 7800)
    np.testing.assert_array_equal(centered_ranks(keys, 60), rankdata(keys) - (len(keys) + 1) / 2)


def test_time_maximum_uses_identical_relabelings_at_each_layer(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from scipy.stats import rankdata
    from phylogeo.verification import dating

    n = 12
    i, j = np.triu_indices(n, 1)
    rng = np.random.default_rng(55)
    lca = np.zeros((n, n), int)
    lca[i, j] = lca[j, i] = np.arange(len(i))
    depth, age = rng.integers(1, 8, size=len(i)), rng.integers(1, 19, size=len(i))
    d = rng.integers(1, 7, (3, n, n))
    d = d + d.swapaxes(1, 2)
    frame = pd.DataFrame({k: ["one"] * n for k in ["stratum", "phylum", "class", "order", "family", "genus"]})
    perms = [rng.permutation(n) for _ in range(19)]
    monkeypatch.setattr(dating, "permutations", lambda *args: iter(perms))
    result = dating.evaluate_time(d, SimpleNamespace(lca=lca, depth=depth), frame, age, len(perms))
    def residual(v):
        rank = rankdata(v)
        design = np.column_stack((np.ones(len(i)), rankdata(depth)))
        return rank - design @ np.linalg.lstsq(design, rank, rcond=None)[0]
    r = np.array([[np.corrcoef(residual(layer[p[i], p[j]]), residual(age))[0, 1] for layer in d]
                  for p in (np.arange(n), *perms)])
    observed, null = abs(r[0]), abs(r[1:]).max(axis=1)
    np.testing.assert_allclose(result["partial"]["by_layer"], r[0], atol=1e-12)
    assert result["partial"]["p_global_max"] == (1 + sum(null >= observed.max())) / (1 + len(null))
    np.testing.assert_allclose(result["partial"]["p_adjusted_by_layer"],
                               [(1 + sum(null >= v)) / (1 + len(null)) for v in observed])


def test_public_decisions_use_their_registered_estimands():
    from phylogeo.verification.publication import assemble
    from phylogeo.verification.statistics import MAIN

    g, d, o, verdict = assemble()
    assert verdict["raw_p"]["S5"] == d["partial"]["p_global_max"]
    assert verdict["raw_p"]["S5"] != d["partial"]["p_fixed_by_layer"][verdict["layer"]]
    assert all(v["historical_max_abs_profile_error"] < 1e-10 for v in g["models"].values())
    assert g["family"][MAIN]["ci"][0] > 1 / 3
    assert g["order"]["pythia410m_step0"]["difference_ci"][0] < 0
    assert g["secondary"]["S2"]["triplets"] < g["secondary"]["S2_broad"]["triplets"]
    assert o["causal"]["expanded"]["registered_holm"]["new_switch"] is None
    assert o["conflict"]["summary"]["char3"]["tie"] > 0
    assert o["habitat"]["holm"]["M2"] is None


def test_zero_weight_neighbour_edges_are_retained():
    from phylogeo.geometry import knn_geodesic
    from phylogeo.verification.graph import graph_distances
    d = np.array([[0., 0, 1], [0, 0, 1], [1, 1, 0]])
    got, k = knn_geodesic(d)
    independent_k, expected = graph_distances(d)
    assert k == independent_k == 1
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got, d)


def test_neighbor_tables_preserve_archived_order_and_all_comparators():
    from phylogeo.config import RESULTS_DIR
    from phylogeo.verification.publication import neighbor_examples
    from phylogeo.verification.statistics import MAIN, read_json

    records = read_json(RESULTS_DIR / MAIN / "examples/neighbors.json")
    full_rows = neighbor_examples().splitlines()[2:]
    compact_rows = neighbor_examples(compact=True).splitlines()[2:]
    comparators = ["tree", *[r for r in records["representations"] if r != "tree"]]
    for query, full, compact in zip(records["queries"], full_rows, compact_rows, strict=True):
        cells = [cell.strip() for cell in full.strip("|").split("|")]
        assert cells == [query["name"], *[", ".join(query["neighbors"][r]) for r in comparators]]
        assert compact.split("|")[2].strip() == query["neighbors"]["tree"][0]


def test_conflict_rows_reproduce_all_choices_and_summary_counts():
    from collections import Counter
    from phylogeo.verification.publication import conflict_items
    from phylogeo.verification.statistics import OUT, read_json

    outcomes = read_json(OUT / "outcomes.json")
    conflict = outcomes["conflict"]
    rows = conflict_items(outcomes).splitlines()[2:]
    comparators = list(conflict["summary"])
    totals = {k: Counter() for k in comparators}
    labels = {1: "Tree", 0: "Tie", -1: "Other", None: "Missing"}
    for record, row in zip(conflict["rows"], rows, strict=True):
        cells = [s.strip() for s in row.strip("|").split("|")]
        assert cells[:4] == [str(record["id"]), record["anchor"], record["closer"], record["other"]]
        for k, cell in zip(comparators, cells[5:], strict=True):
            assert cell == labels[record["choices"][k]]
            totals[k][cell] += 1
    for k, summary in conflict["summary"].items():
        assert [totals[k][v] for v in labels.values()] == [summary[field] for field in ("win", "tie", "loss", "unscorable")]


def test_stage1b_report_keeps_descriptive_tables_and_corrected_decisions():
    from phylogeo.config import RESULTS_DIR
    from phylogeo.report import stage1b_tables
    from phylogeo.verification.publication import assemble, secondary_table
    from phylogeo.verification.statistics import MAIN, read_json

    g, d, _, decisions = assemble()
    text = stage1b_tables(MAIN)
    assert secondary_table(g, d, decisions) in text
    assert f"| No shared character trigram | {g['no_shared_trigram']['triplets']} |" in text
    archived = read_json(RESULTS_DIR / MAIN / "stage1b/results.json")
    for group, values in archived["exploratory"]["per_stratum_ci95"].items():
        assert f"| {group} | {values['n_nodes']} |" in text
