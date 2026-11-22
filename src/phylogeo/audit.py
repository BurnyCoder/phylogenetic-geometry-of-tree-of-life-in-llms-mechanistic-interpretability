"""Independent audit (2026-11-22): re-derive and stress-test headline numbers from stored data.

Global context: phase ``audit`` re-checks numbers reported in README, paper and summary
without any model forward pass or training. It reads stored activations
(``DATA_DIR/activations``) and results JSON, and writes
``results/<run>/audit/results.json``; ``report.py`` renders it and
``reports/reviews/2026-11-22-independent-audit.md`` interprets it. Items:
  1. TA_within recomputed with independent code: every resolved within-stratum,
     head-noun-disjoint triplet enumerated (no per-node cap), LCAs from dendropy
     parent pointers instead of ``RefTree``.
  2. Monte Carlo stability of the H1 paired-bootstrap criterion (5 extra family-block
     seeds) and its sensitivity to the block level (orders instead of families).
  3. The "lexically neutral" post-hoc subset: what it contains, and the untrained
     controls' own CIs on it and on triplets whose names share no token.
  4. Flat probe vs raw angular distance on the *same* held-out pairs (stage 2 compared
     the probe's held-out value with an all-species value).
  5. Variance matching of the stage-3 random subspaces on the held-out species, where
     the interventions act.
  6. Neighbour-joining: fraction of reference splits recovered.
All items are post hoc and exploratory; none changes a registered verdict.
"""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np

from phylogeo.config import ACT_DIR, DATA_REPO_DIR, RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.audit")


def exhaustive_within_triplets(newick: str, leaf_labels: list[str], groups: np.ndarray, heads: np.ndarray,
                               exclude: tuple = ("other",)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All resolved triplets (a, b | c) inside one group with three distinct head nouns.

    Local: LCAs come from dendropy parent pointers (independent of ``RefTree``). A triple is
    resolved when exactly one of its three pairs has a strictly deeper LCA; that pair is (a, b)
    and the decision node is LCA(a, c). Unary nodes do not change these comparisons.
    Returns (a, b, c, decision-node id) arrays.
    """
    import dendropy

    t = dendropy.Tree.get(data=newick, schema="newick", preserve_underscores=True, suppress_internal_node_taxa=True)
    leaf = {nd.taxon.label: nd for nd in t.leaf_node_iter()}
    nid = {id(nd): k for k, nd in enumerate(t.preorder_node_iter())}  # preorder: a parent precedes its children
    depth = {}
    for nd in t.preorder_node_iter():
        depth[nid[id(nd)]] = 0 if nd.parent_node is None else depth[nid[id(nd.parent_node)]] + 1
    chains = []
    for lab in leaf_labels:
        x, ch = leaf[lab], []
        while x is not None:
            ch.append(nid[id(x)])
            x = x.parent_node
        chains.append(ch)
    sets = [set(c) for c in chains]

    def lca(i, j):
        return next(a for a in chains[j] if a in sets[i])

    out = []
    for g in np.unique(groups):
        if g in exclude:
            continue
        for x, y, z in itertools.combinations(np.where(groups == g)[0], 3):
            if len({heads[x], heads[y], heads[z]}) < 3:
                continue
            pairs = {(x, y): lca(x, y), (x, z): lca(x, z), (y, z): lca(y, z)}
            dmax = max(depth[v] for v in pairs.values())
            top = [p for p, v in pairs.items() if depth[v] == dmax]
            if len(top) == 1:
                a, b = top[0]
                c = ({x, y, z} - {a, b}).pop()
                out.append((a, b, c, lca(a, c)))
    arr = np.array(out, dtype=int).reshape(-1, 4)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def node_balanced_ta(D: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, node: np.ndarray) -> float:
    """Mean over decision nodes of the mean triplet score (ties share credit, as registered)."""
    dab, dac, dbc = D[a, b], D[a, c], D[b, c]
    m = np.minimum(np.minimum(dab, dac), dbc)
    k = (dab == m).astype(float) + (dac == m) + (dbc == m)
    s = np.where(dab == m, 1.0 / k, 0.0)
    u, inv = np.unique(node, return_inverse=True)
    return float((np.bincount(inv, weights=s) / np.bincount(inv)).mean())


def run_audit(cfg: RunConfig, load_species, layer_distances, out_dir, write_json) -> dict:
    from transformers import AutoTokenizer

    from phylogeo.analysis import _rng, name_token_counts
    from phylogeo.baselines import char_ngram_distance, scalar_distance
    from phylogeo.behavior import CLASS_WORD
    from phylogeo.causal import random_subspaces
    from phylogeo.extract import load_activations
    from phylogeo.geometry import angular_distance, species_vectors
    from phylogeo.probes.flat import family_folds, per_stratum_spearman
    from phylogeo.stats.triplets import clade_bootstrap_ta, distinct_mask, sample_triplets, ta, within_group_mask
    from phylogeo.treeness import nj_newick

    df, tree = load_species()
    names = df["common_name"].tolist()
    strata, heads = df["stratum"].to_numpy(), df["head_noun"].to_numpy()
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    order = np.where(df["order"] != "", df["order"], fam).astype(object)
    s1b = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())
    s3 = json.loads((RESULTS_DIR / cfg.name / "stage3" / "results.json").read_text())
    main, L = cfg.name, int(s1b["H1"]["modal_layer"])
    untrained = [r for r in s1b["H1"]["required_comparisons"] if r in s1b["models"]]
    D = {r: layer_distances(r)[0] for r in [main] + untrained}
    ml = {r: (L if r == main else int(s1b["models"][r]["modal_layer"])) for r in D}
    res: dict = {"layer": L, "untrained": untrained}

    # 1. Exhaustive, independently coded TA_within.
    nwk = (DATA_REPO_DIR / "species_tree.nwk").read_text(encoding="utf-8").strip()
    a, b, c, v = exhaustive_within_triplets(nwk, [f"ott{o}" for o in df["ott_id"]], strata, heads)
    res["exhaustive"] = {"n_triplets": int(len(a)), "n_decision_nodes": int(len(np.unique(v))),
                         "ta_within_by_layer_main": [node_balanced_ta(Dl, a, b, c, v) for Dl in D[main]],
                         **{f"ta_within_{r}": node_balanced_ta(D[r][ml[r]], a, b, c, v) for r in D},
                         "reported_sampled": {r: s1b["point_ta_within"][r] for r in D}}
    log.info("exhaustive TA_within: %s", {k: x for k, x in res["exhaustive"].items() if k.startswith("ta_within_q")})

    # 2. Monte Carlo stability and block level of the H1 paired bootstrap (same triplets as stage 1b).
    trip = sample_triplets(tree, cfg.k_per_node, _rng(cfg, "triplets"))
    within = within_group_mask(trip, strata) & distinct_mask(trip, heads)
    reps = {main: D[main][L], **{r: D[r][ml[r]] for r in untrained}, "char3": char_ngram_distance(names),
            "input_embedding": D[main][0], "token_count": scalar_distance(name_token_counts(cfg.model_id, names)),
            "log_popularity": scalar_distance(np.log(df["sitelinks"].to_numpy(float)))}
    keys = list(reps)

    def diff_ci(groups, tag):
        bo = clade_bootstrap_ta(list(reps.values()), trip, within, groups, tree, cfg.n_boot, _rng(cfg, tag))
        return {k: np.percentile(bo[0] - bo[i], [2.5, 97.5]).tolist() for i, k in enumerate(keys) if i}

    seeds = [diff_ci(fam, f"audit_boot_{s}") for s in range(5)]
    res["bootstrap_family_seeds"] = {k: [s[k] for s in seeds] for k in keys[1:]}
    res["bootstrap_family_seeds_min_lower"] = {k: float(min(s[k][0] for s in seeds)) for k in keys[1:]}
    res["bootstrap_order_block"] = diff_ci(order, "audit_boot_order")
    res["n_families"], res["n_orders"] = int(len(np.unique(fam))), int(len(np.unique(order)))
    log.info("bootstrap lower bounds (family seeds min): %s; order blocks: %s",
             res["bootstrap_family_seeds_min_lower"], res["bootstrap_order_block"])

    # 3. Lexically neutral subset (char 3-grams tie) and no-shared-token subset, with the controls' own CIs.
    c3 = reps["char3"]
    d1, d2, d3 = c3[trip.a, trip.b], c3[trip.a, trip.c], c3[trip.b, trip.c]
    mn = np.minimum(np.minimum(d1, d2), d3)
    k = (d1 == mn).astype(int) + (d2 == mn) + (d3 == mn)
    neutral = within & (k > 1)
    no3 = within & (k == 3) & (mn == 1.0)

    def ta_ci(mask, tag):
        bo = clade_bootstrap_ta([D[r][ml[r]] for r in D], trip, mask, fam, tree, cfg.n_boot, _rng(cfg, tag))
        return {r: {"ta": float(ta(D[r][ml[r]], trip, mask)), "ci95": np.nanpercentile(bo[i], [2.5, 97.5]).tolist()}
                for i, r in enumerate(D)}

    res["lexically_neutral"] = {"n": int(neutral.sum()), "n_no_shared_char3gram": int(no3.sum()),
                                "n_two_way_tie_wrong_pairs": int((within & (k == 2) & (d1 != mn)).sum()),
                                "by_rep": ta_ci(neutral, "audit_neutral")}
    tok_ids = {}
    for r in D:
        mid = "EleutherAI/pythia-410m" if "pythia" in r else cfg.model_id
        if mid not in tok_ids:
            tk = AutoTokenizer.from_pretrained(mid)
            tok_ids[mid] = [set(tk(" " + x, add_special_tokens=False)["input_ids"]) for x in names]
        ids = tok_ids[mid]
        disj = np.array([not (ids[i] & ids[j] or ids[i] & ids[q] or ids[j] & ids[q])
                         for i, j, q in zip(trip.a, trip.b, trip.c)])
        m = within & disj
        bo = clade_bootstrap_ta([D[r][ml[r]]], trip, m, fam, tree, cfg.n_boot, _rng(cfg, f"audit_tok_{r}"))
        res.setdefault("no_shared_token", {})[r] = {"tokenizer": mid, "n": int(m.sum()), "ta": float(ta(D[r][ml[r]], trip, m)),
                                                    "ci95": np.nanpercentile(bo[0], [2.5, 97.5]).tolist()}

    # 4. Probe vs raw angular distance on the same held-out pairs (stage-2 folds: seed s -> family_folds(seed s)).
    V = species_vectors(load_activations(ACT_DIR / main / "species", "last"))[L]
    A, P = angular_distance(V), tree.path_length()
    s2 = json.loads((RESULTS_DIR / main / "stage2" / "results.json").read_text())
    per_seed = [float(np.mean([per_stratum_spearman(A[np.ix_(te, te)], P[np.ix_(te, te)], strata[te])
                               for te in family_folds(fam, 5, np.random.default_rng(s))]))
                for s in range(int(cfg.extra.get("probe_seeds", 5)))]
    res["probe_vs_raw_heldout"] = {"raw_angular_heldout_within_stratum_spearman": float(np.mean(per_seed)),
                                   "raw_per_seed": per_seed,
                                   "probe_heldout_within_stratum_spearman":
                                       s2["probe"]["model"]["test_test_within_stratum_spearman"][0],
                                   "raw_angular_all_species": s2["probe"]["raw_angular_within_stratum_spearman_all_species"]}

    # 5. Stage-3 random subspaces: captured variance on training vs held-out species.
    has = np.array([s in CLASS_WORD for s in strata])
    f0 = family_folds(fam, 2, np.random.default_rng(cfg.seed))[0]
    train, test = ~f0, f0 & has
    var_tr, var_all = s3["C_abl"]["probe_var_captured"], s3["C_abl_v1"]["probe_var_captured"]
    Urs = random_subspaces(V[train], 10, var_tr, int(s3["C_abl"]["n_random_subspaces"]), np.random.default_rng([cfg.seed, 1]))

    def frac(X, U):
        Xc = X - X.mean(0)
        return float(((Xc @ U) ** 2).sum() / (Xc ** 2).sum())

    tot = lambda X, mu: float(((X - mu) ** 2).sum())
    mu_all = V.mean(0)
    # The probe subspace itself is not stored; its held-out share is bounded from the two stored shares:
    # S_all >= S_train + S_heldout (around the all-species mean), so S_heldout <= var_all*T_all - var_tr*T_train.
    probe_te = (var_all * tot(V, mu_all) - var_tr * tot(V[train], V[train].mean(0))) / tot(V[~train], mu_all)
    res["causal_variance_matching"] = {"probe_train": var_tr, "probe_all_species_v1": var_all,
                                       "probe_heldout_upper_bound": float(probe_te),
                                       "random_train_mean": float(np.mean([frac(V[train], U) for U in Urs])),
                                       "random_heldout_mean": float(np.mean([frac(V[test], U) for U in Urs])),
                                       "random_heldout_range": [float(min(frac(V[test], U) for U in Urs)),
                                                                float(max(frac(V[test], U) for U in Urs))]}

    # 6. Neighbour joining: fraction of the reference tree's non-trivial splits recovered.
    import dendropy

    tns = dendropy.TaxonNamespace()
    n = tree.n

    def splits(newick):
        t = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tns, preserve_underscores=True)
        t.is_rooted = False
        t.encode_bipartitions()
        full = (1 << n) - 1
        return {min(m, full ^ m) for m in (bp.split_bitmask for bp in t.bipartition_encoding)
                if 1 < bin(m).count("1") < n - 1}

    sr = splits(tree.to_newick())
    sn = splits(nj_newick(A, tree.leaf_labels))
    res["nj_splits"] = {"reference_splits": len(sr), "nj_splits": len(sn), "shared": len(sr & sn),
                        "fraction_reference_recovered": len(sr & sn) / len(sr)}
    write_json(out_dir(cfg.name, "audit") / "results.json", res)
    log.info("audit: %s", {k: x for k, x in res.items() if k not in ("exhaustive", "bootstrap_family_seeds")})
    return res
