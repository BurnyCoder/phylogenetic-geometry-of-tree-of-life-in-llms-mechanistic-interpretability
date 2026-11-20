"""Helpers behind the interactive figures: tree ordering, percentiles, relation levels, README gallery."""

import numpy as np

from phylogeo.viz.viz_data import dfs_leaf_order, nearest, percentile_matrix, relation_level


def test_dfs_order_makes_every_clade_contiguous(tree30):
    order = dfs_leaf_order(tree30)
    assert sorted(order) == list(range(tree30.n))
    pos = {leaf: i for i, leaf in enumerate(order)}
    for v in range(tree30.n, tree30.N):  # every internal node's leaves occupy one block of positions
        p = sorted(pos[int(m)] for m in tree30.members[v])
        assert p[-1] - p[0] + 1 == len(p)


def test_percentile_matrix_is_symmetric_and_rank_preserving():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    P = percentile_matrix(D)
    assert P.dtype == np.uint8 and (P == P.T).all() and (np.diag(P) == 0).all()
    iu = np.triu_indices(12, 1)
    assert P[iu].min() == 0 and P[iu].max() == 100
    i, j = np.argsort(D[iu])[[0, -1]]  # closest pair -> 0, farthest pair -> 100
    assert P[iu][i] == 0 and P[iu][j] == 100


def test_relation_level_uses_lowest_shared_rank_and_ignores_blanks():
    tax = {"genus": np.array(["Panthera", "Panthera", "", ""]), "family": np.array(["Felidae"] * 2 + ["", ""]),
           "order": np.array(["Carnivora"] * 3 + ["Primates"]), "class": np.array(["Mammalia"] * 4),
           "phylum": np.array(["Chordata"] * 4)}
    assert relation_level(tax, 0, 1) == 0  # same genus
    assert relation_level(tax, 2, 3) == 3  # blank genus/family are not "shared"; different orders -> same class
    assert relation_level(tax, 0, 2) == 2  # same order


def test_nearest_excludes_self_and_breaks_ties_by_table_order():
    D = np.array([[0, 1, 1, 2], [1, 0, 3, 3], [1, 3, 0, 3], [2, 3, 3, 0]], float)
    nn = nearest(D, 2)
    assert nn[0].tolist() == [1, 2] and 0 not in nn[0]


def test_readme_gallery_links_every_png_to_its_site_card():
    import plotly.graph_objects as go

    from phylogeo.viz.figures import SITE_URL, readme_gallery_md, rec

    figs = [rec(k, f"title {k}", "positive", "ok", "cap", go.Figure()) for k in ["a", "b", "c"]]
    md = readme_gallery_md(figs)
    for k in ["a", "b", "c"]:
        assert f'{SITE_URL}#fig-{k}' in md and f"reports/figures/viz/{k}.png" in md
    assert md.count("<tr>") == 2  # two per row
