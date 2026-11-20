"""Stage 4: ablations of the main analysis (exploratory; preregistered list, section 3).

Global context: each ablation changes ONE choice of the main pipeline and reports
how the headline numbers (TA_within at every layer and at the modal layer,
Mantel r, flat-probe held-out r) move:
  * readout: mean over name tokens vs last name token;
  * distance normalization: raw cosine, per-dimension z-scored, stratum-centered;
  * prompt format: raw text without the chat template (separate extraction run);
  * names: scientific binomials instead of common names (triplets split above
    genus only, since congeners share the genus token);
  * species set: polysemous names included (combined Open Tree tree);
  * probe: dimension k in {2, 3, 5, 10, 20}; angular vs Euclidean vs Poincare
    metric at k in {2, 5, 10}; within-group held-out evaluation;
  * kNN K for the geodesic analysis.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from phylogeo.config import RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.stage4")


def _ta_profile(D, trip, mask):
    from phylogeo.stats.triplets import ta

    return [float(x) for x in np.atleast_1d(ta(D, trip, mask))]


def run_stage4(cfg: RunConfig, load_species, layer_distances, act_dir, out_dir, write_json) -> dict:
    from phylogeo.analysis import _rng, stratum_centered_distances
    from phylogeo.extract import load_activations
    from phylogeo.geometry import angular_distance, cosine_distance, knn_geodesic, smallest_connected_k, species_vectors
    from phylogeo.phases import check_names
    from phylogeo.probes.flat import cv_probe
    from phylogeo.stats.mantel import mantel_r
    from phylogeo.stats.triplets import distinct_mask, sample_triplets, ta, within_group_mask

    df, tree = load_species()
    strata = df["stratum"].to_numpy()
    heads = df["head_noun"].to_numpy()
    genus = df["genus"].to_numpy(dtype=object)
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    trip = sample_triplets(tree, cfg.k_per_node, _rng(cfg, "triplets"))  # same triplets as stage 1b
    within = within_group_mask(trip, strata) & distinct_mask(trip, heads)
    s1b = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())
    L0 = int(s1b["H1"]["modal_layer"])
    P = tree.path_length()
    od = out_dir(cfg.name, "stage4")
    res: dict = {"modal_layer": L0, "variants": {}}

    def add(name, D, mask=within, note=""):
        prof = _ta_profile(D, trip, mask)
        res["variants"][name] = {"ta_within_by_layer": prof, "ta_within_modal": prof[L0] if len(prof) > 1 else prof[0],
                                 "ta_within_max": float(np.nanmax(prof)), "argmax_layer": int(np.nanargmax(prof)),
                                 "mantel_pathlen_modal": float(mantel_r(D[L0] if D.ndim == 3 else D, P)), "note": note}
        log.info("ablation %s: TA_within modal %.3f max %.3f (layer %d)", name, res["variants"][name]["ta_within_modal"],
                 res["variants"][name]["ta_within_max"], res["variants"][name]["argmax_layer"])

    # Readout and normalization variants (main run activations).
    D_main, X = layer_distances(cfg.name)
    add("main (last token, per-template centering)", D_main)
    add("raw cosine (no centering)", layer_distances(cfg.name, variant="raw")[0])
    add("z-scored dimensions", layer_distances(cfg.name, variant="zscore")[0])
    add("stratum-centered", stratum_centered_distances(X, strata))
    add("mean over name tokens", layer_distances(cfg.name, readout="mean")[0])
    del X

    # Prompt format: raw text without the chat template.
    raw_run = cfg.extra.get("raw_text_run", "qwen35_2b_raw")
    if (act_dir(raw_run, "species") / "last.npy").exists():
        add("raw text (no chat template)", layer_distances(raw_run)[0])

    # Scientific binomials: exclude triplets where any two species share a genus.
    if (act_dir(cfg.name, "species_sci") / "last.npy").exists():
        check_names(cfg.name, "species_sci")
        Xs = load_activations(act_dir(cfg.name, "species_sci"), "last")
        Ds = cosine_distance(species_vectors(Xs))
        split = (genus[trip.a] != genus[trip.b]) & (genus[trip.a] != genus[trip.c]) & (genus[trip.b] != genus[trip.c])
        m = within_group_mask(trip, strata) & split
        add("scientific binomials (genus-split triplets)", Ds, m, "common-name TA on the same triplets in note")
        res["variants"]["scientific binomials (genus-split triplets)"]["common_names_same_triplets_modal"] = \
            float(ta(D_main[L0], trip, m))
        del Xs

    # Polysemous names included: combined set and Open Tree tree.
    if (act_dir(cfg.name, "species_polysemous") / "last.npy").exists():
        import pandas as pd

        from phylogeo.config import DATA_REPO_DIR
        from phylogeo.data.opentree import induced_subtree
        from phylogeo.data.tree import RefTree

        dp = pd.read_csv(DATA_REPO_DIR / "species_polysemous.csv", keep_default_na=False)
        comb = pd.concat([df, dp], ignore_index=True)
        nwk, dropped = induced_subtree(comb["ott_id"].tolist())
        if not dropped:
            ct = RefTree(nwk, [f"ott{i}" for i in comb["ott_id"]])
            Xc = np.concatenate([load_activations(act_dir(cfg.name, "species"), "last"),
                                 load_activations(act_dir(cfg.name, "species_polysemous"), "last")])
            Dc = cosine_distance(species_vectors(Xc)[L0])
            tc = sample_triplets(ct, cfg.k_per_node, _rng(cfg, "triplets_polysemous"))
            mc = within_group_mask(tc, comb["stratum"].to_numpy()) & distinct_mask(tc, comb["head_noun"].to_numpy())
            res["polysemous_included"] = {"n_species": len(comb), "ta_within_modal": float(ta(Dc, tc, mc)),
                                          "ta_within_primary_modal": float(ta(D_main[L0], trip, within))}
            log.info("polysemous included: %s", res["polysemous_included"])
            del Xc

    # Probe ablations at the modal layer.
    V = species_vectors(load_activations(act_dir(cfg.name, "species"), "last"))[L0]
    probes = {}
    for k in (2, 3, 5, 10, 20):
        r = [cv_probe(V, P, fam, k=k, seed=s, groups=strata) for s in range(3)]
        probes[f"angular_k{k}"] = {key: float(np.mean([x[key][0] for x in r])) for key in r[0]}
        log.info("probe angular k=%d: %s", k, probes[f"angular_k{k}"])
    for metric in ("euclid", "poincare"):
        for k in (2, 5, 10):
            r = [cv_probe(V, P, fam, k=k, seed=s, metric=metric, groups=strata) for s in range(3)]
            probes[f"{metric}_k{k}"] = {key: float(np.mean([x[key][0] for x in r])) for key in r[0]}
            log.info("probe %s k=%d: %s", metric, k, probes[f"{metric}_k{k}"])
    # Same within-group evaluation for the shuffled-tree control (reference for the within-group number).
    perm = _rng(cfg, "probe_shuffle").permutation(len(df))
    r = [cv_probe(V, P[np.ix_(perm, perm)], fam, k=10, seed=s, groups=strata) for s in range(3)]
    probes["angular_k10_shuffled_tree"] = {key: float(np.mean([x[key][0] for x in r])) for key in r[0]}
    res["probe"] = probes

    # kNN K sweep for the geodesic analysis at the modal layer.
    A = angular_distance(V)
    k0 = smallest_connected_k(A)
    geo = {}
    for k in sorted({k0, 2 * k0, 10, 20, 40}):
        G, _ = knn_geodesic(A, k)
        geo[str(k)] = {"mantel_pathlen": float(mantel_r(G, P)), "ta_within": float(ta(G, trip, within))}
    geo["angular"] = {"mantel_pathlen": float(mantel_r(A, P)), "ta_within": float(ta(A, trip, within))}
    res["geodesic_k_sweep"] = {"smallest_connected_k": k0, "values": geo}
    write_json(od / "results.json", res)
    return res
