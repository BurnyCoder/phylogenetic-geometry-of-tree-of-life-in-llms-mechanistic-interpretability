"""Recheck causal marginals, conflict choices, ecological ranks and published exploratory summaries."""

import hashlib
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, rankdata, spearmanr
from threadpoolctl import threadpool_limits

from phylogeo.config import ACT_DIR, DATA_DIR, REPO_ROOT, RESULTS_DIR
from .primary import spelling, word_baselines
from .statistics import (CONTROLS, MAIN, OUT, FrozenTree, distances, frame, holm_family, log, permutations,
                         read_json, save, stream, tails, transition_bounds)


def intervention(s):
    """Derive both rounds from original observed means. Random draws are subspaces, not species."""
    rounds = {}
    for tag in ("_v1", ""):
        a, w = s["C_abl" + tag], s["C_swap" + tag]
        n = w["n_pairs"]
        before, after = [round(w[k] * n) for k in ("switch_rate_no_edit", "switch_rate_probe")]
        ap = tails(a["mean_drop_logp_probe"], a["random_mean_drops"])
        dp = tails(after / n, w["switch_rate_random"])
        rounds[tag or "expanded"] = {"pairs": n, "before": before, "after": after, "net": after - before,
            "new_donor_bounds": transition_bounds(n, before, after), "ablation_rank": ap,
            "donor_prevalence_rank": dp, "historical_holm": holm_family({"ablation": ap, "prevalence": dp}),
            "registered_holm": holm_family({"ablation": ap, "new_switch": None}),
            "random_prevalence_mean": np.mean(w["switch_rate_random"]), "ablation_probe": a["mean_drop_logp_probe"],
            "ablation_random": np.mean(a["random_mean_drops"]), "random_ablations": len(a["random_mean_drops"]),
            "random_interchanges": len(w["switch_rate_random"])}
    return rounds


def original_subspace_record():
    """Keep the original transcript evidence for the approximate variance match independently replayable."""
    file = REPO_ROOT / "logs/bg6_stage3_causal.out"
    if not file.exists():
        return read_json(OUT / "subspace-transcript.json")
    match = re.search(r"random subspaces from top-(\d+) PCs; mean captured variance ([\d.]+) \(target ([\d.]+)\)",
                      file.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise ValueError("Missing variance-matching measurement in the original intervention transcript")
    record = {"source": file.name, "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
              "line": match.group(), "pc_span": int(match[1]), "random_mean": float(match[2]),
              "target": float(match[3])}
    save("subspace-transcript", record)
    return record


def conflict():
    df = frame("conflict")
    tr = frame("conflict_triplets")
    names = df.common_name.tolist()
    ids = list(dict.fromkeys(df.ott_id))
    tree = FrozenTree([f"ott{k}" for k in ids], "conflict")
    leaf = np.array([ids.index(k) for k in df.ott_id])
    by_name = {v: i for i, v in enumerate(names)}
    l = read_json(RESULTS_DIR / MAIN / "stage1b/results.json")["H1"]["modal_layer"]
    representations = {r: distances(r, names, "conflict")[l] for r in (MAIN, *CONTROLS, "pythia410m")
                       if (ACT_DIR / r / "conflict/last.npy").exists()}
    _, glove, known = word_baselines(names)
    representations.update(char3=spelling(names), glove=glove)
    rows = []
    for rec in tr.itertuples():
        a, p, f = [by_name[v] for v in (rec.anchor, rec.phylo, rec.folk)]
        dp, df_ = tree.depth[tree.lca[leaf[a], leaf[[p, f]]]]
        if dp == df_ or len(set(leaf[[a, p, f]])) < 3:
            rows.append({"id": rec.id, "resolved": False})
            continue
        x, y = (p, f) if dp > df_ else (f, p)
        choices = {k: (None if k == "glove" and not known[[a, x, y]].all() else
                       int(np.sign(d[a, y] - d[a, x]))) for k, d in representations.items()}
        rows.append({"id": rec.id, "anchor": names[a], "closer": names[x], "other": names[y],
                     "resolved": True, "choices": choices})
    summary = {}
    for k in representations:
        codes = [r["choices"][k] for r in rows if r["resolved"]]
        wins, ties, losses = [codes.count(i) for i in (1, 0, -1)]
        summary[k] = {"win": wins, "tie": ties, "loss": losses, "unscorable": codes.count(None),
                      "nominal_p_decisive": binomtest(wins, wins + losses).pvalue if wins + losses else None}
    return {"layer": l, "rows": rows, "summary": summary}


def habitat():
    """Recover the fixed PC axis via a symmetric Gram eigendecomposition on discovery species only."""
    primary, held = frame(), frame("heldout_eco")
    record = read_json(RESULTS_DIR / MAIN / "eco/results.json")
    def layer(key, names):
        directory = ACT_DIR / MAIN / key
        if (directory / "names.txt").read_text(encoding="utf-8").splitlines() != list(names):
            raise ValueError(f"{key} names out of order")
        return np.load(directory / "last.npy", mmap_mode="r")[:, :, record["layer"]].astype(float)
    xp = layer("species", primary.common_name)
    mu = xp.mean(0)
    v = (xp - mu).mean(1)
    _, eig = np.linalg.eigh(v @ v.T)
    axis = v.T @ eig[:, -1]
    axis /= np.linalg.norm(axis)
    discovery = v @ axis
    if discovery[primary.stratum == "birds"].mean() < discovery[primary.stratum == "ray-finned fish"].mean():
        axis *= -1
    s = (layer("heldout_eco", held.common_name) - mu).mean(1) @ axis
    published = {v["common_name"]: v["axis_score"] for v in record["heldout_scores"]}
    if set(published) != set(held.common_name):
        raise ValueError("Held-out habitat scores refer to different species")
    delta = abs(s - [published[n] for n in held.common_name]).max()
    m = held.forstrat_mammal
    ground, marine, aerial = [s[m == k] for k in ("G", "M", "A")]
    birds = held.stratum == "birds"
    bird = held[birds]
    tests = {"M1": mannwhitneyu(marine, ground, alternative="less").pvalue,
             "M2": mannwhitneyu(aerial, ground, alternative="greater").pvalue if len(aerial) >= 5 else None}
    sensitivity, rho = {}, {}
    for key, col, sign in (("B1", "water_pct_bird", -1), ("B2", "aerial_pct_bird", 1)):
        trait = bird[col].astype(float).to_numpy()
        r = spearmanr(s[birds], trait, alternative="less" if sign == -1 else "greater")
        tests[key], rho[key] = r.pvalue, r.statistic
        y, x = rankdata(trait), rankdata(s[birds])
        null = [sign * (x[p] @ y) for p in permutations(bird.order.to_numpy(), 9999, stream("habitat_" + key))]
        sensitivity[key] = tails(sign * (x @ y), null)
    use = m.isin(["M", "G"])
    ranks = rankdata(s[use])
    is_marine = (m[use] == "M").to_numpy()
    null = [-ranks[p][is_marine].sum() for p in permutations(held.order[use].to_numpy(), 9999, stream("habitat_M1"))]
    sensitivity["M1"] = tails(-ranks[is_marine].sum(), null)
    sensitivity["M2"] = None
    files = {"mammals": ("MamFuncDat.txt", "59c3eee29d3ed0a33a002975a5a8cc75"),
             "birds": ("BirdFuncDat.txt", "d6197b2cd90ca3ece0a7393abbf8b7fc")}
    import pandas as pd
    for group, (file, digest) in files.items():
        file = DATA_DIR / "eltontraits" / file
        if hashlib.md5(file.read_bytes()).hexdigest() != digest:
            raise ValueError("Trait table differs from the publisher's reference file")
        source = pd.read_csv(file, sep="\t", encoding="latin-1").set_index("Scientific")
        sub = held[held.stratum == group]
        rows = source.loc[sub.sci_name]
        if group == "mammals":
            np.testing.assert_array_equal(rows["ForStrat-Value"], sub.forstrat_mammal)
        else:
            np.testing.assert_array_equal(rows["ForStrat-watbelowsurf"] + rows["ForStrat-wataroundsurf"],
                                          sub.water_pct_bird.astype(float))
            np.testing.assert_array_equal(rows["ForStrat-aerial"], sub.aerial_pct_bird.astype(float))
    return {"n_heldout": len(held), "birds": int(birds.sum()), "marine": len(marine), "ground": len(ground),
            "aerial": len(aerial), "marine_orders": held.order[m == "M"].value_counts().to_dict(),
            "axis_max_abs_error": delta, "p": tests, "holm": holm_family(tests), "rho": rho,
            "order_conditional_p": sensitivity, "order_conditional_holm": holm_family(sensitivity)}


def behavior(s):
    """Rounded per-question odds survive for behavioral judgments in the original terminal transcript."""
    file = REPO_ROOT / "logs/bg4_stage3_qwen.out"
    n = s["behavior_species_triplets"]["n"]
    if file.exists():
        lines = re.findall(r"triplet \| anchor=(.*?) \| x=(.*?) \| y=(.*?) \| log-odds\(x\)=([\d.\-]+)",
                           file.read_text(encoding="utf-8", errors="replace"))
        if len(lines) != n + s["behavior_conflict"]["n"]:
            raise ValueError("Unexpected number of behavioral log records")
        trials = lines[-n:]
        save("behavior-trials", {"source": file.name, "columns": ["anchor", "x", "y", "rounded_log_odds_x"], "rows": trials})
    else:
        trials = read_json(OUT / "behavior-trials.json")["rows"]
    if len(trials) != n:
        raise ValueError("Unexpected number of species trials")
    odds = np.array([float(x[-1]) for x in trials])
    return {"source": file.name, "n": n, "distinct_anchors": len({r[0] for r in trials}),
            "distinct_species": len({v for r in trials for v in r[:3]}),
            "positive_rounded": int((odds > 0).sum()), "ties_at_log_precision": int((odds == 0).sum()),
            "accuracy": s["behavior_species_triplets"]["behavioral_accuracy"]}


def exploratory():
    """Index stage outputs and check what the archive can identify about the exploratory work."""
    root = RESULTS_DIR / MAIN
    s1a = read_json(root / "stage1a/checks.json")
    s2 = read_json(root / "stage2/results.json")
    s4 = read_json(root / "stage4/results.json")
    s6 = read_json(root / "stage6/results.json")
    power = read_json(RESULTS_DIR / "power/summary.json")["curve"]
    import yaml
    runs = yaml.safe_load((REPO_ROOT / "configs/power.yaml").read_text())["n_sims"]
    for row in power:
        ci = binomtest(round(row["power"] * runs), runs).proportion_ci(method="exact")
        row["binomial_ci"] = [ci.low, ci.high]
    counts = {"power_reps": runs, "power": power, "n_layers_pca": len(s1a),
              "pca_readings": {v: sum(r["reading"] == v for r in s1a) for v in sorted({r["reading"] for r in s1a})},
              "probe": s2["probe"], "gromov": s2["delta_rel"], "nj": s2["nj_rf"],
              "stage4_sections": list(s4), "stage6_sections": list(s6)}
    counts["sensitivity_min"] = {k: s6[k][MAIN]["min"] for k in ("triplet_seed", "species_subsample")}
    counts["sensitivity_min"]["template_half"] = s6["templates"]["half"]["ta_within"]["min"]
    return counts


def run():
    s = read_json(RESULTS_DIR / MAIN / "stage3/results.json")
    with threadpool_limits(limits=1):
        results = {"causal": intervention(s), "subspace_variance": original_subspace_record(),
                   "class_control": s["class_control"], "conflict": conflict(),
                   "behavior": behavior(s), "habitat": habitat(), "exploratory": exploratory()}
    save("outcomes", results)
    log.info("Saved intervention, conflict, habitat and exploratory verification")
