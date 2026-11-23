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
    figs[1]["gallery_img"] = "b_layers.gif"  # an animated record can show a GIF in its gallery cell
    md = readme_gallery_md(figs)
    assert "reports/figures/viz/b_layers.gif" in md and "reports/figures/viz/b.png" not in md


def test_aligned_pca3_keeps_within_layer_geometry_and_removes_axis_flips():
    """Procrustes alignment is display-only: pairwise distances inside a layer are unchanged, and a layer that is
    the previous one with a flipped/rotated basis maps back onto it."""
    from phylogeo.viz.viz_data import aligned_pca3_layers, pca3

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 8)) * np.array([5, 3, 2, 1, 1, 1, 1, 1])  # clear top-3 directions
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    V = np.stack([X, -X @ Q])  # layer 2 = layer 1 in a rotated, reflected basis
    out = aligned_pca3_layers(V)
    C = [np.array(c) for c in out["coords"]]
    raw, _ = pca3(V[1])
    dist = lambda A: np.linalg.norm(A[:, None] - A[None], axis=-1)
    assert np.allclose(dist(C[1]), dist(raw), atol=1e-2)  # rotation only
    assert np.abs(C[1] - C[0]).max() < 2e-2  # aligned back onto the previous layer


def test_unit_group_pca3_ignores_vector_norms():
    """The 3-D within-group view uses unit-length vectors (the geometry cosine distance scores), so rescaling
    each species' vector by any positive factor leaves the picture unchanged."""
    from phylogeo.viz.viz_data import unit_group_pca3

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 16))
    idx = np.arange(5, 35)
    C, evr = unit_group_pca3(X, idx)
    C2, _ = unit_group_pca3(X * rng.uniform(0.1, 10, size=(40, 1)), idx)
    assert C.shape == (30, 3) and len(evr) == 3 and np.allclose(C, C2, atol=1e-6)


def test_layer_sweep_matches_the_static_view_up_to_rotation():
    """The animated layer sweep shows, at every layer, the same 3-D picture as the static view (unit-length,
    group-centred top-3 PCA), only rotated onto the previous layer: within-layer distances are unchanged."""
    from phylogeo.viz.viz_data import euclidean, unit_group_pca3, unit_group_pca3_layers

    rng = np.random.default_rng(5)
    V = rng.normal(size=(3, 40, 16)) * np.array([6, 4, 3] + [1] * 13)  # [layers, species, dims], clear top-3 PCs
    idx = np.arange(10, 40)
    sweep = unit_group_pca3_layers(V, idx)
    assert len(sweep["coords"]) == 3 and np.array(sweep["coords"][0]).shape == (30, 3)
    for lay in range(3):
        static, _ = unit_group_pca3(V[lay], idx)
        assert np.allclose(euclidean(np.array(sweep["coords"][lay])), euclidean(static), atol=5e-3)


def test_within_group_3d_view_keeps_tree_order_of_brownian_motion(tree30):
    """Positive and negative control for the 3-D view: Brownian motion on the tree keeps triplet order above
    chance in 3 dimensions; the same points with shuffled species are at chance (1/3)."""
    from phylogeo.stats.triplets import sample_triplets, ta
    from phylogeo.viz.viz_data import euclidean, unit_group_pca3
    from tests.conftest import brownian_on_tree

    rng = np.random.default_rng(4)
    X = brownian_on_tree(tree30, 64, rng)
    trip = sample_triplets(tree30, 50, rng)
    idx = np.arange(30)
    acc = ta(euclidean(unit_group_pca3(X, idx)[0]), trip)
    acc0 = np.mean([ta(euclidean(unit_group_pca3(X[rng.permutation(30)], idx)[0]), trip) for _ in range(20)])
    # Even a perfect tree code loses part of its order in 3 of 64 dimensions (about 0.58 here), so the bound is
    # "clearly above chance", not near 1.
    assert acc > 0.5 and abs(acc0 - 1 / 3) < 0.08


def test_point_traces_follow_the_given_labels_and_style():
    """Points are grouped and coloured by the labels passed in (here taxonomic orders), not by stratum."""
    from phylogeo.viz.figures import _point_traces

    C = np.zeros((4, 3))
    style = {"Carnivora": ("#2a78d6", "circle"), "other orders": ("#bdbcb6", "circle")}
    tr = _point_traces(C, ["lion", "wolf", "cow", "pig"], ["Carnivora", "Carnivora", "other orders", "other orders"],
                       style, "scene", showlegend=True)
    assert [t.name for t in tr] == ["Carnivora", "other orders"]
    assert tr[0].marker.color == "#2a78d6" and list(tr[0].text) == ["lion", "wolf"]


def test_within3d_readme_block_takes_every_number_from_the_data():
    """The README block next to the H1 highlight is generated: table cells and the summary count come from the
    within3d record, so no hand-typed number can go stale."""
    import plotly.graph_objects as go

    from phylogeo.viz.figures import SITE_URL, rec, within3d_md

    panel = lambda t3, tf, fit: {"ta_3d": t3, "ta_full": tf, "fit_ratio": fit}
    g = {"n_nodes": 43, "diff_ci95_3d": [0.031, 0.266], "diff_ci95_full": [-0.025, 0.291],
         "panels": {"expected": panel(0.67, 0.876, 0.03), "m": panel(0.418, 0.533, 0.33), "c": panel(0.281, 0.398, 0.48)}}
    g["panels"]["m"]["layers"] = {"coords": [[[0, 0, 0]]] * 25}
    w3 = {"main": "m", "control": "c", "hidden_size": 2048, "n_boot": 1000, "layer": 6}
    r = rec("within3d", "title", "mixed", "verdict", "Post hoc, descriptive. caption", go.Figure())
    r.update(within3d=w3, groups=[("mammals", g)], clear={"3d": ["mammals"], "full": []})
    md = within3d_md(r)
    assert f"{SITE_URL}#fig-within3d" in md and "reports/figures/viz/within3d.gif" in md
    assert "reports/figures/viz/within3d_layers.gif" in md and "layers 0 → 24" in md  # both animations are kept
    assert "| mammals (43) | 0.418 / 0.281 [+0.031, +0.266] | 0.533 / 0.398 [-0.025, +0.291] | 0.03 / 0.33 / 0.48 |" in md
    assert "in 1 of 1 groups in the 3-D view and in 0 of 1 in all 2048 dimensions" in md


def test_group_view_caption_states_the_pipeline_with_numbers_from_the_data():
    """The birds caption spells out every pipeline step (readout, centring, unit length, group PCA, chosen angle,
    fitted tree) and takes its counts from the within3d record, the species table and the prompt list."""
    from phylogeo.viz.figures import group_view_caption

    view = lambda t2: {"ta_2d": t2}
    g = {"idx": list(range(70)), "n_nodes": 45,
         "panels": {"expected": {"ta_3d": 0.643, "ta_full": 0.867, "view": view(0.555)},
                    "m": {"layers": {"coords": [[[0, 0, 0]]] * 25}}}}
    w3 = {"main": "m", "control": "c", "hidden_size": 2048, "layer": 6}
    cap = group_view_caption(w3, g, "birds", n_species=400, n_prompts=8)
    assert "birds (70 species, 45 decision nodes)" in cap
    assert "after layer 6 (the output of block 6 of 24" in cap and "in 8 prompts" in cap
    assert "the mean over all 400 species is subtracted, then the prompts are averaged" in cap
    assert "centred on the birds and projected onto the birds' top 3 principal components" in cap
    assert "not the PC1–PC2 plane" in cap and "ancestor nodes" in cap
    assert "0.867 in all 2048 dimensions, 0.643 in 3-D and 0.555 in this view" in cap


def test_label_mask_never_overlaps_names_and_prefers_priority():
    """In the static view, a name is shown only where its text box overlaps no other shown name; when two collide,
    the higher-priority one (lower number, e.g. a highlighted order) wins; isolated points are always labelled."""
    from phylogeo.viz.figures import label_mask

    P = np.array([[0.0, 0.0], [0.001, 0.0], [1.0, 1.0]])  # two points on top of each other, one far away
    keep = label_mask(P, ["osprey", "golden eagle", "mallard"], priority=np.array([1, 0, 1]))
    assert keep.tolist() == [False, True, True]


def test_gif_palette_keeps_series_colours_exact():
    """GIF frames are mapped to a fixed palette, so blue, orange and aqua markers keep their legend colours."""
    from PIL import Image

    from phylogeo.viz.figures import AQUA, BLUE, GRAY, ORANGE, gif_palette

    hexes = [BLUE, ORANGE, AQUA, GRAY]
    px = np.array([[[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes]], np.uint8)
    out = np.array(Image.fromarray(px).quantize(palette=gif_palette(), dither=Image.Dither.NONE).convert("RGB"))
    assert (out == px).all()


def test_shared_box_holds_the_trained_model_and_counts_what_it_hides():
    """The per-layer box contains every trained-model point; points outside it are clipped (NaN) and counted."""
    from phylogeo.viz.figures import _clip, _lim_at, _lims, n_outside

    rng = np.random.default_rng(2)
    trained = rng.normal(size=(3, 50, 3))
    lims = _lims(trained)
    assert all(n_outside(trained[i], lims[i]) == 0 for i in range(3))
    untrained = np.vstack([rng.normal(size=(48, 3)), [[50, 0, 0], [0, -60, 0]]])  # two outlier names
    assert n_outside(untrained, lims[0]) >= 2
    assert np.isnan(_clip(untrained, lims[0])[-2:]).all()
    assert np.isclose(_lim_at(lims, 0.5), (lims[0] + lims[1]) / 2)
