"""Stage 2: manifold / tree geometry (H_geom) and the trained flat-subspace probe.

Global context (preregistration sections 3 and 5, exploratory inference):
  1. Geodesic vs angular: does distance along the kNN-graph manifold track the
     tree better than straight-line angular distance (Goodfire Evo 2; Modell et
     al. 2025)? The gain is compared with the same gain on a class-blob simulation
     (clusters without within-class tree), because kNN graphs on clustered data
     inflate between-cluster distances by themselves.
  2. Gromov delta_rel vs a spectrum-matched Gaussian and the class-blob null.
  3. Neighbor-joining tree from representation distances vs the reference
     (normalized RF) vs NJ trees from label-permuted matrices.
  4. Flat-subspace probe (k = 10) with family-holdout CV, 5 seeds, and controls:
     shuffled-tree target, random-init inputs, GloVe inputs (selectivity).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from phylogeo.config import RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.stage2")


def matched_class_blob(V: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each species = its stratum's real centroid + isotropic noise with the stratum's real spread."""
    out = np.empty_like(V)
    for g in np.unique(strata):
        m = strata == g
        c = V[m].mean(0)
        sd = np.sqrt(((V[m] - c) ** 2).sum(1).mean() / V.shape[1])
        out[m] = c + sd * rng.normal(size=(m.sum(), V.shape[1]))
    return out


def matched_gaussian(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian sample with the same mean and covariance spectrum as V (via SVD)."""
    Vc = V - V.mean(0)
    U, S, Wt = np.linalg.svd(Vc, full_matrices=False)
    return rng.normal(size=U.shape) @ np.diag(S) @ Wt / np.sqrt(1.0)


def run_stage2(cfg: RunConfig, load_species, layer_distances, act_dir, out_dir, write_json) -> dict:
    from phylogeo.baselines import glove_vectors
    from phylogeo.extract import load_activations
    from phylogeo.geometry import angular_distance, cosine_distance, knn_geodesic, species_vectors
    from phylogeo.probes.flat import cv_probe
    from phylogeo.stats.mantel import mantel_r
    from phylogeo.stats.triplets import distinct_mask, sample_triplets, ta, within_group_mask
    from phylogeo.treeness import delta_rel, nj_newick, rf_normalized
    from phylogeo.viz.plots import line_by_layer_png

    rng = np.random.default_rng(cfg.seed)
    df, tree = load_species()
    strata = df["stratum"].to_numpy()
    heads = df["head_noun"].to_numpy()
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    P = tree.path_length()
    trip = sample_triplets(tree, cfg.k_per_node, rng)
    within = within_group_mask(trip, strata) & distinct_mask(trip, heads)
    s1b = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())
    layer = int(cfg.extra.get("stage2_layer", s1b["H1"]["modal_layer"]))
    od = out_dir(cfg.name, "stage2")
    res: dict = {"layer": layer}

    # 1. Geodesic vs angular, all layers, model and matched class-blob simulation.
    V_all = species_vectors(load_activations(act_dir(cfg.name, "species"), "last"))
    geo = {"angular_mantel": [], "geodesic_mantel": [], "angular_ta": [], "geodesic_ta": [], "k": [],
           "blob_gain_mantel": [], "blob_gain_ta": []}
    for l in range(V_all.shape[0]):
        A = angular_distance(V_all[l])
        G, k = knn_geodesic(A)
        B = matched_class_blob(V_all[l], strata, rng)
        Ab = angular_distance(B)
        Gb, _ = knn_geodesic(Ab)
        geo["angular_mantel"].append(float(mantel_r(A, P)))
        geo["geodesic_mantel"].append(float(mantel_r(G, P)))
        geo["angular_ta"].append(float(ta(A, trip, within)))
        geo["geodesic_ta"].append(float(ta(G, trip, within)))
        geo["k"].append(int(k))
        geo["blob_gain_mantel"].append(float(mantel_r(Gb, P) - mantel_r(Ab, P)))
        geo["blob_gain_ta"].append(float(ta(Gb, trip, within) - ta(Ab, trip, within)))
    res["geodesic_vs_angular"] = geo
    line_by_layer_png({"angular": np.array(geo["angular_mantel"]), "kNN geodesic": np.array(geo["geodesic_mantel"])},
                      od / "geodesic_vs_angular_mantel.png", "Mantel Spearman vs tree path length", "Spearman r")
    line_by_layer_png({"angular": np.array(geo["angular_ta"]), "kNN geodesic": np.array(geo["geodesic_ta"])},
                      od / "geodesic_vs_angular_ta.png", "TA_within: geodesic vs angular", "TA_within",
                      hlines={"chance 1/3": 1 / 3})

    # 2. delta_rel at the selected layer vs nulls.
    V = V_all[layer]
    A = angular_distance(V)
    d_model = delta_rel(A, rng=rng)
    d_gauss = [delta_rel(angular_distance(matched_gaussian(V, rng)), rng=rng) for _ in range(10)]
    d_blob = [delta_rel(angular_distance(matched_class_blob(V, strata, rng)), rng=rng) for _ in range(10)]
    res["delta_rel"] = {"model": d_model, "matched_gaussian_mean": float(np.mean(d_gauss)),
                        "matched_gaussian_sd": float(np.std(d_gauss)), "class_blob_mean": float(np.mean(d_blob)),
                        "class_blob_sd": float(np.std(d_blob)), "tree_oracle": float(delta_rel(P, rng=rng))}
    log.info("delta_rel: %s", res["delta_rel"])

    # 3. Neighbor joining vs reference topology.
    labels = tree.leaf_labels
    ref = tree.to_newick()
    rf_model = rf_normalized(nj_newick(A, labels), ref)
    rf_null = []
    for _ in range(20):
        p = rng.permutation(len(labels))
        rf_null.append(rf_normalized(nj_newick(A[np.ix_(p, p)], labels), ref))
    res["nj_rf"] = {"model": rf_model, "perm_null_mean": float(np.mean(rf_null)), "perm_null_sd": float(np.std(rf_null)),
                    "perm_p": float((1 + np.sum(np.array(rf_null) <= rf_model)) / 21)}
    log.info("NJ RF: %s", res["nj_rf"])

    # 4. Flat-subspace probe, 5 seeds + controls, live dashboard (trackio).
    tracker = None
    try:
        import trackio

        tracker = trackio
        trackio.init(project="phylogeo-probes", name=f"{cfg.name}_layer{layer}")
    except Exception as e:  # dashboard is optional; training proceeds without it
        log.warning("trackio unavailable: %s", e)
    seeds = list(range(int(cfg.extra.get("probe_seeds", 5))))
    probe = {}

    def run(tag, X, D):
        out = [cv_probe(X, D, fam, k=10, seed=s, tracker=tracker, tag=f"{tag}/seed{s}") for s in seeds]
        probe[tag] = {key: (float(np.mean([o[key][0] for o in out])), float(np.std([o[key][0] for o in out])))
                      for key in out[0]}
        log.info("probe %s: test-test r = %.3f ± %.3f", tag, *probe[tag]["test_test_pearson"])

    run("model", V, P)
    perm = rng.permutation(len(labels))
    run("shuffled_tree", V, P[np.ix_(perm, perm)])
    ctrl = cfg.extra.get("random_init_run")
    if ctrl and (act_dir(ctrl, "species") / "last.npy").exists():
        Vr = species_vectors(load_activations(act_dir(ctrl, "species"), "last"))
        lr = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())["models"].get(ctrl, {})
        run("random_init", Vr[lr.get("modal_layer", layer)], P)
    gv, _ = glove_vectors(df["common_name"].tolist())
    run("glove", gv, P)
    if tracker is not None:
        trackio.finish()
    probe["selectivity_test_test"] = probe["model"]["test_test_pearson"][0] - probe["shuffled_tree"]["test_test_pearson"][0]
    res["probe"] = probe
    write_json(od / "results.json", res)
    return res
