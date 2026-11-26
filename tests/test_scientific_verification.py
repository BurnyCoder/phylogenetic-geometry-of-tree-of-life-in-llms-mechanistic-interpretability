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
