"""Stage 1b analysis (H1, S1-S5) and the conflict-triplet analysis.

Global context: implements preregistration sections 3-4 on top of the statistics
in ``stats/`` and the baselines in ``baselines.py``. Called from
``phases.phase_analyze`` and ``phases.phase_conflict``. All outputs go to
``results/<run>/stage1b/results.json`` and ``results/<run>/conflict/conflict.json``;
every number reported anywhere is read back from those files.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from phylogeo.config import DATA_DIR, DATA_REPO_DIR, RESULTS_DIR, RunConfig
from phylogeo.data.tree import RefTree

log = logging.getLogger("phylogeo.analysis")
RANK_ORDER = ["phylum", "class", "order", "family", "genus"]


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def pair_rank_level(df: pd.DataFrame) -> np.ndarray:
    """[n, n] deepest shared Linnaean rank index (-1 none, 0 phylum, ..., 4 genus)."""
    n = len(df)
    lvl = np.full((n, n), -1)
    for k, r in enumerate(RANK_ORDER):
        v = df[r].to_numpy(dtype=object)
        same = (v[:, None] == v[None, :]) & (v[:, None] != "")
        lvl[same] = k
    return lvl


def _group_perm(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        p[idx] = rng.permutation(idx)
    return p


def s5_rank_vs_time(D: np.ndarray, ages: np.ndarray, tree: RefTree, lvl: np.ndarray, groups: np.ndarray,
                    n_perm: int, rng: np.random.Generator) -> dict:
    """S5: pooled within-rank-bin Spearman(rep distance, TimeTree MRCA age), permutation null.

    Pairs are binned by the deepest shared Linnaean rank; within each bin (>= 30
    pairs, >1 distinct age) the Spearman correlation between representation
    distance and MRCA age is computed; bins are pooled weighted by pair count.
    Null: species permuted within strata (same as H1). Two-sided p.
    """
    from scipy.stats import rankdata

    iu = np.triu_indices(len(groups), 1)
    age = ages[tree.lca][iu]
    lv = lvl[iu]
    bins = [b for b in np.unique(lv) if (lv == b).sum() >= 30 and np.unique(age[lv == b]).size > 1]
    ranked_age = {b: rankdata(age[lv == b]) for b in bins}

    def pooled(d):
        num, den, per = 0.0, 0, {}
        for b in bins:
            m = lv == b
            r = np.corrcoef(ranked_age[b], rankdata(d[m]))[0, 1]
            per[int(b)] = (float(r), int(m.sum()))
            num += r * m.sum()
            den += m.sum()
        return num / den, per

    obs, per = pooled(D[iu])
    null = np.array([pooled(D[np.ix_(p, p)][iu])[0] for p in (_group_perm(groups, rng) for _ in range(n_perm))])
    pval = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return {"pooled_spearman": float(obs), "p_two_sided": float(pval), "maxT_p": float(pval),
            "per_bin": {(RANK_ORDER[b] if b >= 0 else "none"): v for b, v in per.items()}}


def run_stage1b(cfg: RunConfig, load_species, layer_distances, act_dir, out_dir, write_json) -> dict:
    """Compute H1, S1-S5, depth profile, baselines and controls; write results + figures."""
    from phylogeo.baselines import char_ngram_distance, class_oracle, glove_vectors, rank_oracle
    from phylogeo.geometry import cosine_distance, split_half_reliability
    from phylogeo.stats.mantel import kendall_tau_b, mantel_maxT, mantel_r
    from phylogeo.stats.triplets import (clade_bootstrap_ta, crossfit_ta, discordant_mask, distinct_mask,
                                         maxT_ta, sample_triplets, shared_rank_level, ta, within_group_mask)
    from phylogeo.viz.plots import line_by_layer_png

    rng = np.random.default_rng(cfg.seed)
    df, tree = load_species()
    names = df["common_name"].tolist()
    strata = df["stratum"].to_numpy()
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    heads = df["head_noun"].to_numpy()
    ranks = {r: df[r].to_numpy(dtype=object) for r in RANK_ORDER}
    od = out_dir(cfg.name, "stage1b")
    trip = sample_triplets(tree, cfg.k_per_node, rng)
    within = within_group_mask(trip, strata) & distinct_mask(trip, heads)
    shared = shared_rank_level(trip.a, trip.b, trip.c, ranks, RANK_ORDER)
    in_order = np.isin(shared, ["order", "family", "genus"])
    log.info("triplets: %d total, %d within-stratum head-disjoint, %d within-order",
             len(trip), int(within.sum()), int(in_order.sum()))

    # Baselines (single distance matrices).
    char3 = char_ngram_distance(names)
    gv, gcov = glove_vectors(names)
    glove = cosine_distance(gv - gv.mean(0))
    base = {"char3": char3, "glove": glove, "class_oracle": class_oracle(strata),
            "rank_oracle": rank_oracle(ranks), "tree_oracle": tree.lca_depth_distance()}
    s3 = within & discordant_mask(char3, trip)
    s4 = within & discordant_mask(glove, trip)
    log.info("S3 lexical-discordant: %d ; S4 co-occurrence-discordant: %d", int(s3.sum()), int(s4.sum()))

    # Main model + untrained / other controls.
    runs = [cfg.name] + [r for r in cfg.extra.get("controls", []) if (act_dir(r, "species") / "last.npy").exists()]
    model_res, D_by_run, modal = {}, {}, {}
    for r in runs:
        D, X = layer_distances(r)
        D_by_run[r] = D
        obs, p, _ = maxT_ta(D, trip, within, strata, cfg.n_perm, rng)
        cf, layers = crossfit_ta(D, trip, within, fam, int(cfg.extra.get("n_splits", 20)), rng)
        ml = int(np.bincount(layers).argmax())
        modal[r] = ml
        m_obs, m_p, _ = mantel_maxT(D, tree.path_length(), cfg.n_perm, rng)
        rec = {"ta_within_by_layer": obs, "maxT_p": p, "crossfit_ta_within": cf, "crossfit_layers": layers,
               "modal_layer": ml, "ta_all_by_layer": ta(D, trip),
               "mantel_pathlen_by_layer": m_obs, "mantel_pathlen_maxT_p": m_p,
               "mantel_lcadepth_by_layer": mantel_r(D, tree.lca_depth_distance()),
               "kendall_tau_b_pathlen_modal": kendall_tau_b(D[ml], tree.path_length()),
               "ta_order_by_layer": ta(D, trip, in_order),
               "ta_s3_by_layer": ta(D, trip, s3), "ta_s4_by_layer": ta(D, trip, s4)}
        if r == cfg.name:
            rec["noise_ceiling_by_layer"] = split_half_reliability(X, rng)
            base["input_embedding"] = D[0]
        model_res[r] = rec
        log.info("%s: TA_within max %.3f (layer %d), max-T p=%.4f, cross-fit %.3f (modal layer %d); "
                 "Mantel(pathlen) max %.3f p=%.4f", r, np.nanmax(obs), int(np.nanargmax(obs)), p, cf, ml,
                 np.max(m_obs), m_p)
        del X

    main = cfg.name
    Dm = D_by_run[main][modal[main]]
    # Paired clade-block bootstrap across all representations (same family draws).
    reps = {main: Dm, **{r: D_by_run[r][modal[r]] for r in runs[1:]}, **base}
    boots = clade_bootstrap_ta(list(reps.values()), trip, within, fam, tree, cfg.n_boot, rng)
    keys = list(reps)
    point = {k: float(ta(v, trip, within)) for k, v in reps.items()}
    ci = {k: np.percentile(boots[i], [2.5, 97.5]).tolist() for i, k in enumerate(keys)}
    diffs = {k: np.percentile(boots[0] - boots[i], [2.5, 97.5]).tolist() for i, k in enumerate(keys) if i > 0}

    # H1 decision (preregistration 4.2).
    req = [k for k in keys if k in runs[1:] or k in ("char3", "input_embedding")]
    h1 = {"maxT_p": model_res[main]["maxT_p"], "crossfit_ta_within": model_res[main]["crossfit_ta_within"],
          "modal_layer": modal[main], "ta_within_modal": point[main], "ci95": ci[main],
          "diff_ci95_vs": {k: diffs[k] for k in req}}
    crit = {"maxT_p<0.05": h1["maxT_p"] < 0.05, "ci_lower>1/3": ci[main][0] > 1 / 3,
            "beats_untrained_and_lexical": all(diffs[k][0] > 0 for k in req)}
    h1["criteria"] = crit
    h1["supported"] = bool(all(crit.values()))
    h1["meaningful_effect_ge_0.40"] = bool(h1["crossfit_ta_within"] >= 0.40)

    # Secondary hypotheses S1-S5 (Holm across the family).
    sec = {"S1": {"mantel_spearman_pathlen_modal": float(model_res[main]["mantel_pathlen_by_layer"][modal[main]]),
                  "maxT_p": model_res[main]["mantel_pathlen_maxT_p"],
                  "kendall_tau_b": model_res[main]["kendall_tau_b_pathlen_modal"]}}
    for key, mask in (("S2", in_order), ("S3", s3), ("S4", s4)):
        obs, p, _ = maxT_ta(D_by_run[main], trip, mask, strata, cfg.n_perm, rng)
        sec[key] = {"ta_by_layer": obs, "maxT_p": p, "ta_modal": float(obs[modal[main]]),
                    "n_triplets": int(mask.sum())}
    ages_path = DATA_DIR / "timetree" / "species_node_ages.npy"
    if ages_path.exists():
        sec["S5"] = s5_rank_vs_time(Dm, np.load(ages_path), tree, pair_rank_level(df), strata, cfg.n_perm, rng)
    hp = holm({k: v["maxT_p"] for k, v in sec.items()})
    for k in sec:
        sec[k]["holm_p"] = hp[k]

    # Depth profile (descriptive) and per-stratum TA.
    levels = ["none"] + RANK_ORDER
    depth = {name: {lv: (float(ta(Dx, trip, shared == lv)) if (shared == lv).any() else None) for lv in levels}
             for name, Dx in reps.items()}
    per_stratum = {s: float(ta(Dm, trip, within & (strata[trip.a] == s))) for s in np.unique(strata)
                   if (within & (strata[trip.a] == s)).any()}
    res = {"n_species": len(df), "n_triplets": len(trip), "n_within": int(within.sum()), "glove_coverage": gcov,
           "H1": h1, "secondary": sec, "point_ta_within": point, "ci95_ta_within": ci, "paired_diff_ci95": diffs,
           "models": model_res, "depth_profile": depth,
           "depth_counts": {lv: int((shared == lv).sum()) for lv in levels},
           "per_stratum_ta_within_modal": per_stratum,
           "baselines_mantel_pathlen": {k: float(mantel_r(v, tree.path_length())) for k, v in base.items()}}
    write_json(od / "results.json", res)

    series = {r: np.asarray(model_res[r]["ta_within_by_layer"]) for r in runs}
    hl = {"chance 1/3": 1 / 3, **{f"{k} (baseline)": point[k] for k in ("char3", "glove", "class_oracle")}}
    line_by_layer_png(series, od / "ta_within_by_layer.png",
                      "Within-stratum, head-noun-disjoint rooted-triplet accuracy", "TA_within", hlines=hl)
    line_by_layer_png({r: np.asarray(model_res[r]["mantel_pathlen_by_layer"]) for r in runs},
                      od / "mantel_by_layer.png", "Mantel Spearman: cosine distance vs tree path length", "Spearman r")
    log.info("H1: %s", json.dumps(h1, default=float))
    return res


def run_conflict(cfg: RunConfig, load_conflict, layer_distances, act_dir, out_dir, write_json) -> dict:
    """Phylogeny-vs-folk conflict triplets: does the closer representation follow the tree?"""
    from scipy.stats import binomtest

    from phylogeo.baselines import char_ngram_distance, glove_vectors
    from phylogeo.geometry import cosine_distance

    cdf, ctree = load_conflict()
    trip = pd.read_csv(DATA_REPO_DIR / "conflict_triplets.csv")
    idx = {n: i for i, n in enumerate(cdf["common_name"])}
    names = cdf["common_name"].tolist()
    reps = {}
    for r in [cfg.name] + cfg.extra.get("controls", []):
        if (act_dir(r, "conflict") / "last.npy").exists():
            reps[r] = layer_distances(r, "conflict")[0]
    gv, _ = glove_vectors(names)
    reps["char3"] = char_ngram_distance(names)[None]
    reps["glove"] = cosine_distance(gv - gv.mean(0))[None]
    leaf_of = {i: (ctree.leaf_labels.index(f"ott{o}") if f"ott{o}" in ctree.leaf_labels else None)
               for i, o in enumerate(cdf["ott_id"])}
    rows = []
    for _, t in trip.iterrows():
        a, p, f = idx[t.anchor], idx[t.phylo], idx[t.folk]
        la, lp, lf = leaf_of[a], leaf_of[p], leaf_of[f]
        if None in (la, lp, lf) or len({la, lp, lf}) < 3:
            rows.append({"id": int(t.id), "resolved": False})
            continue
        dp, dfk = ctree.depth[ctree.lca[la, lp]], ctree.depth[ctree.lca[la, lf]]
        if dp == dfk:
            rows.append({"id": int(t.id), "resolved": False})
            continue
        closer, other = (p, f) if dp > dfk else (f, p)
        row = {"id": int(t.id), "anchor": t.anchor, "tree_closer": names[closer], "other": names[other],
               "matches_prereg_phylo": bool(closer == p), "type": t.conflict_type, "resolved": True}
        for r, D in reps.items():
            row[r] = (D[:, a, closer] < D[:, a, other]).astype(int).tolist()
        rows.append(row)
    ok = [x for x in rows if x["resolved"]]
    layer = int(cfg.extra.get("conflict_layer", -1))
    if layer < 0:
        s1b = RESULTS_DIR / cfg.name / "stage1b" / "results.json"
        layer = json.loads(s1b.read_text())["H1"]["modal_layer"] if s1b.exists() else 12
    summary = {}
    for r, D in reps.items():
        l = layer if D.shape[0] > 1 else 0
        k = int(sum(x[r][l] for x in ok))
        summary[r] = {"layer": l, "n": len(ok), "phylo_choice": k, "accuracy": k / len(ok),
                      "binom_p_two_sided": float(binomtest(k, len(ok), 0.5).pvalue),
                      "accuracy_by_layer": [sum(x[r][ll] for x in ok) / len(ok) for ll in range(D.shape[0])]}
        log.info("conflict %s: %d/%d choose the tree-closer partner (layer %d)", r, k, len(ok), l)
    out = {"summary": summary, "triplets": rows}
    write_json(out_dir(cfg.name, "conflict") / "conflict.json", out)
    return out
