"""Output examples: nearest neighbours of fixed query species under each representation.

Global context: the paper and site show qualitative examples next to the
statistics. To avoid cherry-picking, the queries are fixed by a rule set before
looking at neighbours: the most-linked species (Wikidata sitelinks) of every
stratum. For each query the 3 nearest other species are listed under the model
(stage-1b modal layer), the untrained controls, character 3-grams, GloVe and the
reference tree (closest relatives by deepest shared ancestor; ties by path
length). A descriptive summary over all species is added: the fraction of each
species' 5 nearest neighbours that share its taxonomic order (NCBI-first ranks).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from phylogeo.config import RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.examples")


def nearest(D: np.ndarray, i: int, k: int) -> list[int]:
    """Indices of the k nearest other items to i under distance matrix D (stable order)."""
    d = D[i].astype(float).copy()
    d[i] = np.inf
    return [int(j) for j in np.argsort(d, kind="stable")[:k]]


def same_label_rate(D: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Mean fraction of each item's k nearest neighbours that share its label (items with a label only)."""
    ok = np.array([bool(x) for x in labels])
    vals = [np.mean([labels[j] == labels[i] for j in nearest(D, i, k)]) for i in np.where(ok)[0]]
    return float(np.mean(vals))


def run_examples(cfg: RunConfig, load_species, layer_distances, act_dir, out_dir, write_json) -> dict:
    from phylogeo.baselines import char_ngram_distance, glove_vectors
    from phylogeo.geometry import cosine_distance

    df, tree = load_species()
    names = df["common_name"].tolist()
    strata = df["stratum"].to_numpy()
    s1b = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())
    reps = {}
    for r in [cfg.name, cfg.extra.get("random_init_run", "qwen35_2b_random"), "pythia410m_step0", "pythia410m"]:
        if (act_dir(r, "species") / "last.npy").exists():
            D, X = layer_distances(r)
            reps[r] = D[int(s1b["models"][r]["modal_layer"])]
            del D, X
    reps["char3"] = char_ngram_distance(names)
    gv, gok = glove_vectors(names)
    reps["glove"] = cosine_distance(gv - gv[gok].mean(0))
    # Tree: deepest shared ancestor first (LCA-depth distance), path length breaks ties.
    P = tree.path_length()
    reps["tree"] = tree.lca_depth_distance() + P / (P.max() + 1)
    pop = df["sitelinks"].to_numpy(float)
    queries = [int(np.where(strata == s)[0][np.argmax(pop[strata == s])]) for s in sorted(set(strata))]
    out = {"layer": int(s1b["H1"]["modal_layer"]), "queries": [], "representations": list(reps)}
    for q in queries:
        out["queries"].append({"name": names[q], "stratum": strata[q],
                               "neighbors": {r: [names[j] for j in nearest(D, q, 3)] for r, D in reps.items()}})
    order = df["order"].fillna("").to_numpy(dtype=object) if "order" in df else np.array([""] * len(df))
    out["same_order_rate_5nn"] = {r: same_label_rate(D, order, 5) for r, D in reps.items()}
    write_json(out_dir(cfg.name, "examples") / "neighbors.json", out)
    log.info("5-NN same-order rate: %s", {k: round(v, 3) for k, v in out["same_order_rate_5nn"].items()})
    return out
