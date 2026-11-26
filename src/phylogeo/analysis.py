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
    """S5: does distance track divergence *time* beyond rank and topology?

    Pairs are binned by the deepest shared Linnaean rank. Within each bin (>= 30
    pairs, >1 distinct age) we compute the partial Spearman correlation between
    representation distance and TimeTree MRCA age **controlling for the
    topological depth of the MRCA** (review finding 6: within bins, age and LCA
    depth correlate at -0.6 to -0.99, so a raw correlation cannot separate time
    from topology). Bins are pooled weighted by pair count; the raw (non-partial)
    pooled Spearman is reported too. Pairs whose MRCA is undated (NaN) are
    excluded. Null: species permuted within strata (respects the dependence of
    pairwise data; the registered "within rank bins" wording is not well defined
    for species-level permutations, see Deviations). Two-sided p on the partial.
    """
    from scipy.stats import rankdata

    iu = np.triu_indices(len(groups), 1)
    age = ages[tree.lca][iu]
    depth = tree.depth[tree.lca][iu].astype(float)
    lv = lvl[iu].copy()
    n_undated = int(np.isnan(age).sum())
    lv[np.isnan(age)] = -99
    bins = [b for b in np.unique(lv) if b != -99 and (lv == b).sum() >= 30 and np.unique(age[lv == b]).size > 1]
    ra = {b: rankdata(age[lv == b]) for b in bins}
    rt = {b: rankdata(depth[lv == b]) for b in bins}

    def corr(x, y):
        c = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(c) else c

    def pooled(d):
        num_p, num_r, den, per = 0.0, 0.0, 0, {}
        for b in bins:
            m = lv == b
            rd = rankdata(d[m])
            r_da, r_dt, r_at = corr(rd, ra[b]), corr(rd, rt[b]), corr(ra[b], rt[b])
            denom = np.sqrt(max((1 - r_dt**2) * (1 - r_at**2), 1e-12))
            r_p = (r_da - r_dt * r_at) / denom
            per[int(b)] = {"partial": float(r_p), "raw": float(r_da), "age_vs_lca_depth": float(r_at),
                           "n_pairs": int(m.sum())}
            num_p += r_p * m.sum()
            num_r += r_da * m.sum()
            den += m.sum()
        return num_p / den, num_r / den, per

    obs, raw, per = pooled(D[iu])
    null = np.array([pooled(D[np.ix_(p, p)][iu])[0] for p in (_group_perm(groups, rng) for _ in range(n_perm))])
    pval = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return {"pooled_partial_spearman": float(obs), "pooled_raw_spearman": float(raw), "p_two_sided": float(pval),
            "fixed_layer_p": float(pval), "n_pairs_undated_excluded": n_undated,
            "per_bin": {(RANK_ORDER[b] if b >= 0 else "none"): v for b, v in per.items()}}


def _rng(cfg: RunConfig, component: str) -> np.random.Generator:
    """Independent, reproducible random stream per analysis component (review finding 15)."""
    import zlib

    return np.random.default_rng([cfg.seed, zlib.crc32(component.encode())])


def name_token_counts(model_id: str, names: list[str]) -> np.ndarray:
    """Number of tokens of " name" under the model's tokenizer (nuisance variable)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return np.array([len(tok(" " + n, add_special_tokens=False)["input_ids"]) for n in names])


def stratum_centered_distances(X: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Cosine distances after subtracting each stratum's mean species vector (per layer).

    Robustness variant (review finding 2): removes between-stratum offsets and the
    norm heterogeneity that global centering can turn into spurious within-stratum
    structure.
    """
    from phylogeo.geometry import cosine_distance, species_vectors

    V = species_vectors(X)  # [L, n, d]
    for g in np.unique(strata):
        m = strata == g
        V[:, m] -= V[:, m].mean(1, keepdims=True)
    return cosine_distance(V)


def run_stage1b(cfg: RunConfig, load_species, layer_distances, act_dir, out_dir, write_json) -> dict:
    """Compute H1, S1-S5, depth profile, baselines and controls; write results + figures.

    Implements preregistration sections 3-4 with the pre-data deviations 11-17
    (review fixes): fixed required control set, nuisance baselines (name token
    count, log popularity) added to the H1 comparisons, head-noun-disjoint S2,
    NaN-aware and topology-controlled S5, NCBI-first ranks, a stratum-centered
    robustness variant, nuisance-discordant triplets, and independent RNG streams.
    """
    from phylogeo.baselines import (char_ngram_distance, class_oracle, glove_vectors, rank_oracle,
                                    scalar_distance)
    from phylogeo.geometry import cosine_distance, split_half_reliability
    from phylogeo.stats.mantel import kendall_tau_b, mantel_maxT, mantel_r
    from phylogeo.stats.triplets import (clade_bootstrap_ta, crossfit_ta, discordant_mask, distinct_mask,
                                         maxT_ta, sample_triplets, shared_rank_level, ta, within_group_mask)
    from phylogeo.viz.plots import line_by_layer_png

    df, tree = load_species()
    names = df["common_name"].tolist()
    strata = df["stratum"].to_numpy()
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    heads = df["head_noun"].to_numpy()
    ranks = {r: df[r].to_numpy(dtype=object) for r in RANK_ORDER}
    od = out_dir(cfg.name, "stage1b")
    trip = sample_triplets(tree, cfg.k_per_node, _rng(cfg, "triplets"))
    hd = distinct_mask(trip, heads)
    within = within_group_mask(trip, strata) & hd
    shared = shared_rank_level(trip.a, trip.b, trip.c, ranks, RANK_ORDER)
    a, b, c = trip.a, trip.b, trip.c
    order = ranks["order"]
    same_order = (order[a] != "") & (order[a] == order[b]) & (order[b] == order[c])
    fine_split = np.logical_or.reduce([(v[a] != "") & (v[c] != "") & (v[a] == v[b]) & (v[b] != v[c])
                                       for v in (ranks["family"], ranks["genus"])])
    in_order = same_order & hd & fine_split
    log.info("triplets: %d total, %d within-stratum head-disjoint, %d within-order head-disjoint",
             len(trip), int(within.sum()), int(in_order.sum()))

    # Baselines (single distance matrices).
    char3 = char_ngram_distance(names)
    gv, gok = glove_vectors(names)
    glove = cosine_distance(gv - gv[gok].mean(0))
    gvalid = gok[trip.a] & gok[trip.b] & gok[trip.c]
    ntok = name_token_counts(cfg.model_id, names)
    logpop = np.log(df["sitelinks"].to_numpy(float))
    base = {"char3": char3, "glove": glove, "token_count": scalar_distance(ntok),
            "log_popularity": scalar_distance(logpop), "class_oracle": class_oracle(strata),
            "rank_oracle": rank_oracle(ranks), "tree_oracle": tree.lca_depth_distance()}
    s3 = within & discordant_mask(char3, trip)
    s4 = within & gvalid & discordant_mask(glove, trip)
    nuis = within & discordant_mask(base["token_count"], trip) & discordant_mask(base["log_popularity"], trip)
    log.info("S3 lexical-discordant: %d ; S4 co-occurrence-discordant: %d ; nuisance-discordant: %d",
             int(s3.sum()), int(s4.sum()), int(nuis.sum()))

    # Required H1 comparisons: fixed set (deviation 11). Missing -> H1 not evaluable.
    rand_run = cfg.extra.get("random_init_run", "qwen35_2b_random")
    required_models = [rand_run, "pythia410m_step0"]
    required = required_models + ["char3", "input_embedding", "token_count", "log_popularity"]
    runs = [cfg.name] + [r for r in cfg.extra.get("controls", []) if (act_dir(r, "species") / "last.npy").exists()]
    missing = [r for r in required_models if r not in runs]

    model_res, D_by_run, modal = {}, {}, {}
    for r in runs:
        D, X = layer_distances(r)
        D_by_run[r] = D
        obs, p, _ = maxT_ta(D, trip, within, strata, cfg.n_perm, _rng(cfg, f"maxT_{r}"))
        cf, layers = crossfit_ta(D, trip, within, fam, int(cfg.extra.get("n_splits", 20)), _rng(cfg, f"cf_{r}"))
        ml = int(np.bincount(layers).argmax())
        modal[r] = ml
        m_obs, m_p, _ = mantel_maxT(D, tree.path_length(), cfg.n_perm, _rng(cfg, f"mantel_{r}"))
        rec = {"ta_within_by_layer": obs, "maxT_p": p, "crossfit_ta_within": cf, "crossfit_layers": layers,
               "modal_layer": ml, "ta_all_by_layer": ta(D, trip),
               "mantel_pathlen_by_layer": m_obs, "mantel_pathlen_maxT_p": m_p,
               "mantel_lcadepth_by_layer": mantel_r(D, tree.lca_depth_distance()),
               "kendall_tau_b_pathlen_modal": kendall_tau_b(D[ml], tree.path_length()),
               "ta_order_by_layer": ta(D, trip, in_order), "ta_s3_by_layer": ta(D, trip, s3),
               "ta_s4_by_layer": ta(D, trip, s4), "ta_nuisance_discordant_by_layer": ta(D, trip, nuis)}
        if r == cfg.name:
            rec["noise_ceiling_by_layer"] = split_half_reliability(X, _rng(cfg, "noise_ceiling"))
            base["input_embedding"] = D[0]
            Dsc = stratum_centered_distances(X, strata)
            o2, p2, _ = maxT_ta(Dsc, trip, within, strata, cfg.n_perm, _rng(cfg, "maxT_stratum_centered"))
            rec["stratum_centered"] = {"ta_within_by_layer": o2, "maxT_p": p2, "ta_within_modal": float(o2[ml])}
        model_res[r] = rec
        log.info("%s: TA_within max %.3f (layer %d), max-T p=%.4f, cross-fit %.3f (modal layer %d); "
                 "Mantel(pathlen) max %.3f p=%.4f", r, np.nanmax(obs), int(np.nanargmax(obs)), p, cf, ml,
                 np.max(m_obs), m_p)
        del X

    main = cfg.name
    Dm = D_by_run[main][modal[main]]
    # Paired clade-block bootstrap across all representations (same family draws).
    reps = {main: Dm, **{r: D_by_run[r][modal[r]] for r in runs[1:]}, **base}
    keys = list(reps)
    boots = clade_bootstrap_ta(list(reps.values()), trip, within, fam, tree, cfg.n_boot, _rng(cfg, "bootstrap"))
    point = {k: float(ta(v, trip, within)) for k, v in reps.items()}
    ci = {k: np.percentile(boots[i], [2.5, 97.5]).tolist() for i, k in enumerate(keys)}
    diffs = {k: np.percentile(boots[0] - boots[i], [2.5, 97.5]).tolist() for i, k in enumerate(keys) if i > 0}
    point["glove_known_names_only"] = float(ta(glove, trip, within & gvalid))

    # H1 decision (preregistration 4.2 + deviations 11-12).
    h1 = {"maxT_p": model_res[main]["maxT_p"], "crossfit_ta_within": model_res[main]["crossfit_ta_within"],
          "modal_layer": modal[main], "ta_within_modal": point[main], "ci95": ci[main],
          "required_comparisons": required, "missing_controls": missing,
          "diff_ci95_vs": {k: diffs[k] for k in required if k in diffs}}
    crit = {"maxT_p<0.05": h1["maxT_p"] < 0.05, "ci_lower>1/3": ci[main][0] > 1 / 3,
            "beats_required_controls": all(diffs[k][0] > 0 for k in required if k in diffs)}
    h1["criteria"] = crit
    h1["supported"] = None if missing else bool(all(crit.values()))
    # Both estimates are reported; the "meaningful" label uses the more conservative one
    # (cross-fit avoids layer-selection optimism but is evaluated on fold subsets; review finding 7).
    h1["ta_within_conservative"] = float(min(h1["crossfit_ta_within"], h1["ta_within_modal"]))
    h1["meaningful_effect_ge_0.40"] = bool(h1["ta_within_conservative"] >= 0.40)

    # Secondary hypotheses S1-S5 (Holm across the family of 5; S5 required).
    sec = {"S1": {"mantel_spearman_pathlen_modal": float(model_res[main]["mantel_pathlen_by_layer"][modal[main]]),
                  "maxT_p": model_res[main]["mantel_pathlen_maxT_p"],
                  "kendall_tau_b": model_res[main]["kendall_tau_b_pathlen_modal"]}}
    for key, mask in (("S2", in_order), ("S3", s3), ("S4", s4)):
        obs, p, _ = maxT_ta(D_by_run[main], trip, mask, strata, cfg.n_perm, _rng(cfg, f"maxT_{key}"))
        sec[key] = {"ta_by_layer": obs, "maxT_p": p, "ta_modal": float(obs[modal[main]]),
                    "n_triplets": int(mask.sum())}
    ages_path = DATA_DIR / "timetree" / "species_node_ages.npy"
    if not ages_path.exists():
        raise FileNotFoundError("S5 needs TimeTree node ages: run the 'dates' phase first")
    ages = np.load(ages_path)
    if len(ages) != tree.N:
        raise ValueError("node ages do not match the current tree: rerun the 'dates' phase")
    sec["S5"] = s5_rank_vs_time(Dm, ages, tree, pair_rank_level(df), strata, cfg.n_perm, _rng(cfg, "S5"))
    from phylogeo.verification.dating import evaluate_time

    time_curve = evaluate_time(D_by_run[main], tree, df, ages, draws=cfg.n_perm)
    sec["S5"]["maxT_p"] = time_curve["partial"]["p_global_max"]
    sec["S5"]["layer_profiles"] = time_curve
    hp = holm({k: v["maxT_p"] for k, v in sec.items()})
    for k in sec:
        sec[k]["holm_p"] = hp[k]

    # Exploratory: nuisance-discordant triplets and stratum-centered robustness at the modal layer.
    expl = {"nuisance_discordant": {"n_triplets": int(nuis.sum()),
                                    "ta_modal": float(model_res[main]["ta_nuisance_discordant_by_layer"][modal[main]])},
            "stratum_centered": model_res[main]["stratum_centered"]}
    # Post hoc (final review round 1; labelled exploratory): compare the model with the untrained
    # controls on the secondary subsets, on lexically neutral triplets (character 3-grams have no
    # unique closest pair), and per stratum, all with the paired family-block bootstrap.
    untrained = [r for r in required_models if r in D_by_run]
    c3 = char3
    d1, d2, d3 = c3[trip.a, trip.b], c3[trip.a, trip.c], c3[trip.b, trip.c]
    mn = np.minimum(np.minimum(d1, d2), d3)
    char3_tie = within & (((d1 == mn).astype(int) + (d2 == mn) + (d3 == mn)) > 1)
    Ds_cmp = [Dm] + [D_by_run[r][modal[r]] for r in untrained]

    def paired(mask, tag):
        bo = clade_bootstrap_ta(Ds_cmp, trip, mask, fam, tree, cfg.n_boot, _rng(cfg, f"posthoc_{tag}"))
        rec = {"n_triplets": int(mask.sum()), main: float(ta(Dm, trip, mask))}
        for i, r in enumerate(untrained, start=1):
            rec[r] = float(ta(Ds_cmp[i], trip, mask))
            rec[f"diff_ci95_vs_{r}"] = np.nanpercentile(bo[0] - bo[i], [2.5, 97.5]).tolist()
        rec[f"{main}_ci95"] = np.nanpercentile(bo[0], [2.5, 97.5]).tolist()
        return rec

    expl["posthoc_vs_untrained"] = {"S2_within_order": paired(in_order, "S2"), "S3_lexical_discordant": paired(s3, "S3"),
                                    "S4_cooccurrence_discordant": paired(s4, "S4"),
                                    "char3_tie_lexically_neutral": paired(char3_tie, "char3tie")}
    expl["per_stratum_ci95"] = {}
    for s_ in np.unique(strata):
        m_ = within & (strata[trip.a] == s_)
        if m_.any():
            bo = clade_bootstrap_ta([Dm], trip, m_, fam, tree, cfg.n_boot, _rng(cfg, f"stratum_{s_}"))
            expl["per_stratum_ci95"][s_] = {"ta": float(ta(Dm, trip, m_)), "ci95": np.nanpercentile(bo[0], [2.5, 97.5]).tolist(),
                                           "n_nodes": int(len(np.unique(trip.node[m_])))}
    strongest = max(required, key=lambda k: point.get(k, -1))
    expl["margin_vs_strongest_required"] = {"control": strongest, "margin": point[main] - point[strongest],
                                            "diff_ci95": diffs.get(strongest)}
    # Depth profile (descriptive) and per-stratum TA.
    levels = ["none"] + RANK_ORDER
    depth = {name: {lv: (float(ta(Dx, trip, shared == lv)) if (shared == lv).any() else None) for lv in levels}
             for name, Dx in reps.items()}
    per_stratum = {s: float(ta(Dm, trip, within & (strata[trip.a] == s))) for s in np.unique(strata)
                   if (within & (strata[trip.a] == s)).any()}
    res = {"n_species": len(df), "n_triplets": len(trip), "n_within": int(within.sum()),
           "glove_coverage": float(gok.mean()), "H1": h1, "secondary": sec, "exploratory": expl,
           "point_ta_within": point, "ci95_ta_within": ci, "paired_diff_ci95": diffs,
           "models": model_res, "depth_profile": depth,
           "depth_counts": {lv: int((shared == lv).sum()) for lv in levels},
           "per_stratum_ta_within_modal": per_stratum,
           "baselines_mantel_pathlen": {k: float(mantel_r(v, tree.path_length())) for k, v in base.items()}}
    write_json(od / "results.json", res)

    series = {r: np.asarray(model_res[r]["ta_within_by_layer"]) for r in runs}
    series[f"{main} (stratum-centered)"] = np.asarray(model_res[main]["stratum_centered"]["ta_within_by_layer"])
    hl = {"chance 1/3": 1 / 3, **{f"{k}": point[k] for k in ("char3", "token_count", "log_popularity")}}
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
    gv, glove_known = glove_vectors(names)
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
            delta = D[:, a, other] - D[:, a, closer]
            row[r] = (np.sign(delta) / 2 + .5).tolist()  # 1 tree, 0 other, half a point for a tie
            if r == "glove" and not glove_known[[a, closer, other]].all():
                row[r] = [None]
        rows.append(row)
    ok = [x for x in rows if x["resolved"]]
    layer = int(cfg.extra.get("conflict_layer", -1))
    if layer < 0:
        s1b = RESULTS_DIR / cfg.name / "stage1b" / "results.json"
        layer = json.loads(s1b.read_text())["H1"]["modal_layer"] if s1b.exists() else 12
    summary = {}
    for r, D in reps.items():
        l = layer if D.shape[0] > 1 else 0
        vals = [x[r][l] for x in ok if x[r][l] is not None]
        k, losses = vals.count(1), vals.count(0)
        summary[r] = {"layer": l, "n": len(vals), "phylo_choice": k, "other_choice": losses,
                      "ties": vals.count(.5), "unscorable": len(ok) - len(vals),
                      "accuracy": float(np.mean(vals)) if vals else None,
                      "binom_p_two_sided": float(binomtest(k, k + losses, .5).pvalue) if k + losses else None,
                      "accuracy_by_layer": []}
        for ll in range(D.shape[0]):
            choices = [x[r][ll] for x in ok if x[r][ll] is not None]
            summary[r]["accuracy_by_layer"].append(float(np.mean(choices)) if choices else None)
        log.info("conflict %s: %d/%d choose the tree-closer partner (layer %d)", r, k, len(ok), l)
    # Does the trained model differ from the controls on the same triplets? Exact two-sided McNemar
    # on the discordant triplets (final review M4; triplets share anchors, so this is descriptive).
    lm = summary[cfg.name]["layer"]
    for r in reps:
        if r == cfg.name:
            continue
        lr_ = summary[r]["layer"]
        b_ = sum(x[cfg.name][lm] == 1 and x[r][lr_] == 0 for x in ok)
        c_ = sum(x[cfg.name][lm] == 0 and x[r][lr_] == 1 for x in ok)
        summary[r]["mcnemar_vs_main"] = {"main_only": int(b_), "control_only": int(c_),
                                         "p_two_sided": float(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0}
    out = {"summary": summary, "triplets": rows}
    write_json(out_dir(cfg.name, "conflict") / "conflict.json", out)
    return out
