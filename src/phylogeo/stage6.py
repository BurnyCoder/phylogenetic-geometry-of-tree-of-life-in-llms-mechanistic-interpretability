"""Stage 6: sensitivity of the headline numbers to seeds, templates, species and splits.

Global context (planned run R6-sens): every headline statistic of stages 1b and 2
was computed once, with fixed seeds and the full template and species sets. Here
each source of arbitrariness is varied on its own, with everything else held at
the main-run values, and the spread is reported (mean, SD, 2.5/97.5 percentiles,
min, max). All statistics are at the stage-1b modal layer unless stated.
  - Triplet sampling: 20 seeds; triplets per decision node K in {50, 100, 200, 400}.
  - Prompt templates: 20 random halves (4 of 8), the 8 leave-one-out sets, and
    each single template. Also records which layer maximizes TA_within.
  - Species: 20 random 80% subsamples (tree re-induced, triplets re-sampled).
  - Cross-fitting: 20 seeds of the family split (each with 20 splits).
  - Flat probe: 20 seeds (each sets both the family folds and the initialization).
The untrained controls (random-init Qwen, Pythia step0) and the nuisance baselines
are recomputed under the triplet and species perturbations, so the margin of the
model over the strongest control is reported for every repetition.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from phylogeo.config import RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.stage6")


def summarize(x) -> dict:
    """Spread of a list of repetitions (SD with ddof=1)."""
    x = np.asarray(x, float)
    return {"values": x, "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "p2.5": float(np.percentile(x, 2.5)), "p97.5": float(np.percentile(x, 97.5)),
            "min": float(x.min()), "max": float(x.max())}


def _per_ref(vals: dict[str, list], main: str) -> dict:
    """Summaries for every representation plus the per-repetition margin over the best control."""
    out = {k: summarize(v) for k, v in vals.items()}
    ctrl = np.max([vals[k] for k in vals if k != main], axis=0)
    out["margin_vs_best_control"] = summarize(np.asarray(vals[main]) - ctrl)
    out["best_control_by_rep"] = [max((k for k in vals if k != main), key=lambda k: vals[k][i])
                                  for i in range(len(vals[main]))]
    return out


def run_stage6(cfg: RunConfig, load_species, layer_distances, act_dir, out_dir, write_json) -> dict:
    from phylogeo.analysis import _rng, name_token_counts
    from phylogeo.baselines import char_ngram_distance, scalar_distance
    from phylogeo.geometry import cosine_distance, species_vectors
    from phylogeo.probes.flat import cv_probe
    from phylogeo.stats.mantel import mantel_r
    from phylogeo.stats.triplets import crossfit_ta, distinct_mask, sample_triplets, ta, within_group_mask
    from phylogeo.viz.plots import strip_png

    df, tree = load_species()
    names = df["common_name"].tolist()
    strata = df["stratum"].to_numpy()
    heads = df["head_noun"].to_numpy()
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    s1b = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())
    L = int(s1b["H1"]["modal_layer"])
    n_rep = int(cfg.extra.get("n_sensitivity", 20))
    od = out_dir(cfg.name, "stage6")
    P = tree.path_length()
    main = cfg.name

    D_main, X = layer_distances(main)  # [layers, n, n] and [n, T, layers, d]
    refs = {main: D_main[L]}
    for r in (cfg.extra.get("random_init_run", "qwen35_2b_random"), "pythia410m_step0"):
        if (act_dir(r, "species") / "last.npy").exists():
            Dr, Xr = layer_distances(r)
            refs[r] = Dr[int(s1b["models"][r]["modal_layer"])]  # each control at its own modal layer
            del Dr, Xr
    refs |= {"char3": char_ngram_distance(names),
             "token_count": scalar_distance(name_token_counts(cfg.model_id, names)),
             "log_popularity": scalar_distance(np.log(df["sitelinks"].to_numpy(float)))}
    res: dict = {"layer": L, "n_rep": n_rep, "representations": list(refs)}

    def within(trip, idx=None):
        s, h = (strata, heads) if idx is None else (strata[idx], heads[idx])
        return within_group_mask(trip, s) & distinct_mask(trip, h)

    trip0 = sample_triplets(tree, cfg.k_per_node, _rng(cfg, "triplets"))  # the stage-1b triplets
    w0 = within(trip0)
    res["main_run"] = {k: float(ta(D, trip0, w0)) for k, D in refs.items()}

    # 1. Triplet sampling seed and triplets per node.
    vals = {k: [] for k in refs}
    for s in range(n_rep):
        trip = sample_triplets(tree, cfg.k_per_node, _rng(cfg, f"s6_triplets_{s}"))
        w = within(trip)
        for k, D in refs.items():
            vals[k].append(float(ta(D, trip, w)))
    res["triplet_seed"] = _per_ref(vals, main)
    res["k_per_node"] = {}
    for K in (50, 100, 200, 400):
        trip = sample_triplets(tree, K, _rng(cfg, f"s6_k{K}"))
        w = within(trip)
        res["k_per_node"][str(K)] = {"n_within": int(w.sum()), **{k: float(ta(D, trip, w)) for k, D in refs.items()}}
    log.info("triplet seed: %s", {k: round(v["mean"], 3) for k, v in res["triplet_seed"].items() if "mean" in v})

    # 2. Prompt templates (model only; activations of the other templates are dropped).
    T = X.shape[1]

    def tpl(sub):
        D = cosine_distance(species_vectors(X[:, sub]))
        prof = np.atleast_1d(ta(D, trip0, w0))
        return float(prof[L]), int(np.nanargmax(prof)), float(mantel_r(D[L], P))

    sets = {"half": [np.sort(_rng(cfg, f"s6_tpl_{s}").choice(T, T // 2, replace=False)) for s in range(n_rep)],
            "leave_one_out": [np.delete(np.arange(T), t) for t in range(T)],
            "single": [np.array([t]) for t in range(T)]}
    res["templates"] = {}
    for key, subs in sets.items():
        out = [tpl(sub) for sub in subs]
        res["templates"][key] = {"subsets": [s.tolist() for s in subs], "ta_within": summarize([o[0] for o in out]),
                                 "argmax_layer": [o[1] for o in out], "mantel_pathlen": summarize([o[2] for o in out])}
        log.info("templates %s: TA_within %.3f ± %.3f", key, res["templates"][key]["ta_within"]["mean"],
                 res["templates"][key]["ta_within"]["sd"])

    # 3. Species subsamples (80%), tree re-induced and triplets re-sampled.
    n, m = len(df), int(round(0.8 * len(df)))
    vals, mant = {k: [] for k in refs}, []
    for s in range(n_rep):
        r = _rng(cfg, f"s6_species_{s}")
        keep = np.sort(r.choice(n, m, replace=False))
        sub = tree.subtree(keep.tolist())  # leaf order = keep order
        trip = sample_triplets(sub, cfg.k_per_node, r)
        w = within(trip, keep)
        for k, D in refs.items():
            vals[k].append(float(ta(D[np.ix_(keep, keep)], trip, w)))
        mant.append(float(mantel_r(D_main[L][np.ix_(keep, keep)], sub.path_length())))
    res["species_subsample"] = _per_ref(vals, main) | {"mantel_pathlen_model": summarize(mant), "n_keep": m}
    log.info("species 80%%: TA_within %.3f ± %.3f", res["species_subsample"][main]["mean"],
             res["species_subsample"][main]["sd"])

    # 4. Family-split seeds of the cross-fitted TA (layer chosen out of fold).
    cf, ml = [], []
    for s in range(n_rep):
        v, layers = crossfit_ta(D_main, trip0, w0, fam, int(cfg.extra.get("n_splits", 20)), _rng(cfg, f"s6_cf_{s}"))
        cf.append(v)
        ml.append(int(np.bincount(layers).argmax()))
    res["crossfit_seed"] = {"crossfit_ta_within": summarize(cf), "modal_layer": ml}

    # 5. Flat-probe seeds (fresh seeds 1000+, disjoint from the stage-2 seeds 0-4).
    V = species_vectors(X)[L]
    del X
    pr = [cv_probe(V, P, fam, k=10, seed=1000 + s, groups=strata) for s in range(n_rep)]
    res["probe_seed"] = {"test_test_pearson": summarize([p["test_test_pearson"][0] for p in pr]),
                         "test_test_within_group_spearman": summarize([p["test_test_within_group_spearman"][0]
                                                                       for p in pr]),
                         "test_test_within_stratum_spearman": summarize([p["test_test_within_stratum_spearman"][0]
                                                                         for p in pr])}
    log.info("probe seeds: held-out r %.3f ± %.3f", res["probe_seed"]["test_test_pearson"]["mean"],
             res["probe_seed"]["test_test_pearson"]["sd"])
    write_json(od / "results.json", res)

    # Figure: TA_within under each perturbation, model vs the strongest untrained/nuisance control.
    best = lambda blk: [max(blk[k]["values"][i] for k in refs if k != main) for i in range(n_rep)]
    rows = {"triplet sampling seed": {main: res["triplet_seed"][main]["values"],
                                      "best control (same draw)": best(res["triplet_seed"])},
            "species 80% subsample": {main: res["species_subsample"][main]["values"],
                                      "best control (same draw)": best(res["species_subsample"])},
            "template halves (4 of 8)": {main: res["templates"]["half"]["ta_within"]["values"]},
            "single template": {main: res["templates"]["single"]["ta_within"]["values"]},
            "cross-fit family splits": {main: res["crossfit_seed"]["crossfit_ta_within"]["values"]}}
    strip_png(rows, od / "sensitivity_ta_within.png",
              f"Sensitivity of TA_within (layer {L}); one dot per repetition", "TA_within",
              vlines={"chance 1/3": 1 / 3, "main run": res["main_run"][main]})
    return res
