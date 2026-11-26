"""Rebuild primary and secondary geometry statistics directly from names, trees, and saved arrays."""

import itertools
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer

from .statistics import (CONTROLS, MAIN, FrozenTree, bootstrap, check_freeze, crossfit, distances, frame,
                         log, node_weights, pair_scores, permutations, read_json, save, scores, stream,
                         tails, triplet_test)


def spelling(names):
    grams = []
    for name in names:
        padded = " " + name.lower() + " "
        grams.append({padded[i:i+3] for i in range(len(padded)-2)})
    return np.array([[1 - len(a & b) / len(a | b) for b in grams] for a in grams])


def word_baselines(names):
    """Use only the published static word table and the pinned tokenizer vocabulary, offline."""
    cache = Path(os.environ["HF_HOME"]) / "hub"
    qwen = cache / "models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
    tok = Tokenizer.from_file(str(qwen / "tokenizer.json"))
    counts = np.array([len(tok.encode(" " + name, add_special_tokens=False).ids) for name in names])
    glove = (cache / "models--sentence-transformers--average_word_embeddings_glove.6B.300d/snapshots"
             / "e5e8fec6971be8960cfaa853a77a6ddc62a265d7/0_WordEmbeddings")
    vocab = {w: i for i, w in enumerate(read_json(glove / "whitespacetokenizer_config.json")["vocab"])}
    items = [n.lower().replace("-", " ").replace("'s", " ").replace("'", " ").split() for n in names]
    with safe_open(glove / "model.safetensors", framework="numpy") as f:
        table = f.get_slice("emb_layer.weight")
        dimension = table.get_shape()[1]
        words = {w: table[vocab[w]:vocab[w]+1][0] for ws in items for w in set(ws) if w in vocab}
    known = np.array([any(w in words for w in ws) for ws in items])
    vectors = np.array([np.mean([words[w] for w in ws if w in words], axis=0)
                        if ok else np.full(dimension, np.nan) for ws, ok in zip(items, known)])
    vectors[~known] = np.nanmean(vectors, axis=0)
    vectors -= vectors[known].mean(axis=0)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return counts, np.clip(1 - vectors @ vectors.T, 0, 2), known


def discordance(d, t):
    a, b, c = t[:3]
    ab, ac, bc = d[a, b], d[a, c], d[b, c]
    return ((ac < ab) & (ac < bc)) | ((bc < ab) & (bc < ac))


def mantel(d, tree, layer):
    """Mantel statistic from rank vectors; permute whole species identically at every layer."""
    i, j = np.triu_indices(tree.n, 1)
    ref = np.zeros_like(tree.path, dtype=float)
    ref[i, j] = rankdata(tree.path[i, j])
    ref += ref.T.copy()
    ranks = rankdata(d[:, i, j], axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    ranks /= np.linalg.norm(ranks, axis=1, keepdims=True)
    ref = (ref - ref[i, j].mean()) / np.linalg.norm(ref[i, j] - ref[i, j].mean())
    obs = ranks @ ref[i, j]
    null = [np.max(ranks @ ref[p[i], p[j]]) for p in permutations(np.zeros(tree.n), 1000, stream("mantel_" + MAIN))]
    return {"by_layer": obs, "modal": float(obs[layer]), "p": tails(obs.max(), null)}


def all_eligible(tree, heads, groups, d):
    """Reference enumeration independent of the capped sampling implementation."""
    rows = []
    for g in np.unique(groups):
        if g == "other":
            continue
        indices = np.flatnonzero(groups == g)
        if len(indices) < 3:
            continue
        i, j, k = np.array(list(itertools.combinations(indices, 3))).T
        good = (heads[i] != heads[j]) & (heads[i] != heads[k]) & (heads[j] != heads[k])
        for a, b, c in ((i, j, k), (i, k, j), (j, k, i)):
            node = tree.lca[a, c]
            take = good & (node == tree.lca[b, c]) & (tree.depth[tree.lca[a, b]] > tree.depth[node])
            rows.append(np.stack((a[take], b[take], c[take], node[take])))
    t = np.concatenate(rows, axis=1)
    return {"n_triplets": t.shape[1], "n_nodes": len(np.unique(t[3])),
            "ta": float(scores(d, t) @ node_weights(t[3]))}


def comparison(ds, t, labels, tree, tag, historical=False, *, distinct_blocks=False):
    keys = list(ds)
    s = np.stack([scores(d, t, historical) for d in ds.values()])
    points = s @ node_weights(t[3])
    boots = bootstrap(s, t, labels, tree, tag, distinct_blocks=distinct_blocks)
    return {k: {"ta": points[i], "ci": np.quantile(boots[i], [.025, .975]),
                "difference": points[0] - points[i],
                "difference_ci": np.quantile(boots[0] - boots[i], [.025, .975])} for i, k in enumerate(keys)}


def verify():
    df = frame()
    names = df.common_name.tolist()
    tree = FrozenTree([f"ott{k}" for k in df.ott_id])
    t = tree.triplets()
    a, b, c, _ = t
    groups, heads = df.stratum.to_numpy(), df.head_noun.to_numpy()
    families = np.where(df.family != "", df.family, df.genus)
    orders = df.order.to_numpy()
    same = (groups[a] != "other") & (groups[a] == groups[b]) & (groups[a] == groups[c])
    head = (heads[a] != heads[b]) & (heads[a] != heads[c]) & (heads[b] != heads[c])
    within = same & head
    ts = t[:, within]
    seq, glove, known = word_baselines(names)
    chars = spelling(names)
    popularity = np.log(df.sitelinks.to_numpy())
    archived = read_json(Path(__file__).resolve().parents[3] / "results/qwen35_2b/stage1b/results.json")
    results = {"frozen": check_freeze(), "sample": {"total": len(a), "eligible": ts.shape[1],
               "nodes": len(np.unique(ts[3])), "families": len(np.unique(families)), "orders": len(np.unique(orders))},
               "models": {}, "secondary": {}}
    selected, old_selected = {}, {}
    for run in (MAIN, *CONTROLS):
        log.info("Checking frozen representations and layer selection for %s", run)
        ds = distances(run, names)
        if run == MAIN:
            main_d = ds
        historical = scores(ds, ts, historical=True)
        old_cf, old_layer = crossfit(historical, ts, families, stream("cf_" + run))
        history = historical @ node_weights(ts[3])
        old_selected[run] = ds[old_layer]
        cf, layer = crossfit(scores(ds, ts), ts, families, stream("cf_" + run))
        selected[run] = ds[layer]
        test = triplet_test(ds, ts, groups, "maxT_" + run)
        results["models"][run] = {**test, "crossfit": cf, "layer": layer, "historical_cf": old_cf,
                                   "historical_layer": old_layer, "historical_by_layer": history,
                                   "historical_max_abs_profile_error": np.max(abs(history - archived["models"][run]["ta_within_by_layer"]))}
    layer = results["models"][MAIN]["layer"]
    refs = {"char3": chars, "input_embedding": main_d[0], "token_count": abs(seq[:, None] - seq),
            "log_popularity": abs(popularity[:, None] - popularity)}
    old_selected.update(refs)
    selected.update(refs)
    results["historical_family"] = comparison(old_selected, ts, families, tree, "bootstrap", True)
    results["family"] = comparison(selected, ts, families, tree, "bootstrap")
    results["distinct_family"] = comparison(selected, ts, families, tree, "bootstrap", distinct_blocks=True)
    results["order"] = comparison(selected, ts, orders, tree, "verification_order")
    results["census"] = all_eligible(tree, heads, groups, main_d[layer])
    results["transported_heads"] = triplet_test(main_d, t[:, same], groups, "maxT_" + MAIN, heads=heads)
    tests = results["secondary"]
    tests["S1"] = mantel(main_d, tree, layer)
    f, g = df.family.to_numpy(), df.genus.to_numpy()
    broad = (orders[a] != "") & (orders[a] == orders[b]) & (orders[a] == orders[c]) & head
    split = ((f[a] != "") & (f[c] != "") & (f[a] == f[b]) & (f[a] != f[c])) | (
             (g[a] != "") & (g[c] != "") & (g[a] == g[b]) & (g[a] != g[c]))
    valid = known[a] & known[b] & known[c]
    masks = {"S2": broad & split, "S2_broad": broad, "S3": within & discordance(chars, t),
             "S4": within & valid & discordance(glove, t)}
    for key, keep in masks.items():
        log.info("Checking %s on %d triplets", key, keep.sum())
        test = triplet_test(main_d, t[:, keep], groups, "maxT_" + key[:2])
        tests[key] = {**test, "modal": test["by_layer"][layer], "triplets": keep.sum(),
                      "nodes": len(np.unique(t[3, keep])),
                      "comparisons": comparison({k: selected[k] for k in (MAIN, *CONTROLS)},
                                                t[:, keep], families, tree, "posthoc_" + key)}
    gram_null = within & (chars[a, b] == 1) & (chars[a, c] == 1) & (chars[b, c] == 1)
    results["no_shared_trigram"] = {"triplets": gram_null.sum(), "comparison": comparison(
        {k: selected[k] for k in (MAIN, *CONTROLS)}, t[:, gram_null], families, tree, "verification_no_grams")}
    results["glove_known"] = known.sum()
    # A stronger sensitivity null preserves the taxonomy of the order for every permuted name.
    tests["S2"]["within_order_permutation_p"] = triplet_test(
        main_d, t[:, masks["S2"]], orders, "verification_order_S2")["p"]
    results["raw_probe_comparison"] = raw_heldout(main_d[layer], tree.path, families, groups)
    save("geometry", results)


def raw_heldout(d, path, families, groups):
    """Recover the probe study's folds and equal-stratum raw-distance comparator, without any probe."""
    values = []
    for seed in range(5):
        shuffled = np.random.default_rng(seed).permutation(np.unique(families))
        for fold in range(5):
            heldout = np.isin(families, shuffled[fold::5])
            correlations = []
            for group in np.unique(groups[heldout]):
                inds = np.flatnonzero(heldout & (groups == group))
                if len(inds) < 3:
                    continue
                i, j = np.triu_indices(len(inds), 1)
                target = path[inds[i], inds[j]]
                if np.ptp(target):
                    correlations.append(spearmanr(d[inds[i], inds[j]], target).statistic)
            values.append(np.mean(correlations))
    return {"per_fold": values, "mean": np.mean(values)}


def run():
    with threadpool_limits(limits=1):
        verify()


if __name__ == "__main__":
    from phylogeo.config import setup_logging

    setup_logging("verify_geometry")
    run()
