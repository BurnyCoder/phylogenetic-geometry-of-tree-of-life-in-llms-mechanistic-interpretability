"""Render Markdown tables from results JSON (no hand-typed numbers anywhere).

Global context: the ``report`` phase regenerates ``results/RESULTS.md`` (and
per-stage tables) purely from ``results/<run>/**/*.json``. README, paper and
experiment notes link to or embed these generated tables, so every reported
number is traceable to a results file and one command rebuilds them all:
``uv run python -m phylogeo.run --phases report``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phylogeo.config import RESULTS_DIR


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{nd}f}"
    return str(x)


def p_fmt(p: float) -> str:
    """Report the value itself; a permutation p at its floor 1/(n+1) shows as e.g. 0.001, never "<"."""
    return f"{p:.1e}" if p < 0.0005 else f"{p:.3f}"


def exploratory_bh(run: str) -> dict:
    """BH-FDR (q = 0.05) over the enumerated exploratory outcomes that have reported p-values (preregistration
    section 3; final review M7). Writes results/<run>/exploratory_bh.json and returns it.

    Family: H1a checks (C1, C2 per zoom stratum, C3, C3b at every layer), NJ vs random trees and vs
    class blobs, conflict triplets (representation and behavior), behavioral species triplets,
    stage-3 class-only control tests, and the max-T p of every stage-5 model.
    This reproduces the historical reporting family. Choice of exploratory tests and dependence limit inference.
    """
    from scipy.stats import false_discovery_control

    R = RESULTS_DIR / run
    tests: dict[str, float] = {}
    for l, c in enumerate(_load(R / "stage1a" / "checks.json") or []):
        tests[f"H1a C1 layer {l}"] = c["C1_p"]
        for s, v in c["C2_within_stratum_order_purity"].items():
            tests[f"H1a C2 {s} layer {l}"] = v["p"]
        tests[f"H1a C3 layer {l}"], tests[f"H1a C3b layer {l}"] = c["C3_p"], c["C3b_p"]
    s2 = _load(R / "stage2" / "results.json")
    if s2:
        tests["NJ RF vs random trees"] = s2["nj_rf"]["perm_p"]
        if "class_blob_p" in s2["nj_rf"]:
            tests["NJ RF vs class-blob trees"] = s2["nj_rf"]["class_blob_p"]
    c = _load(R / "conflict" / "conflict.json")
    if c:
        tests["conflict triplets (representation)"] = c["summary"][run]["binom_p_two_sided"]
    s3 = _load(R / "stage3" / "results.json")
    if s3:
        tests["conflict triplets (behavior)"] = s3["behavior_conflict"]["binom_p"]
        tests["species triplets (behavior)"] = s3["behavior_species_triplets"]["binom_p"]
        if "class_control" in s3:
            tests["C-abl probe > class-only subspace"] = s3["class_control"]["wilcoxon_p_probe_gt_class"]
            tests["C-swap probe > class-only subspace"] = s3["class_control"]["mcnemar_p_probe_gt_class"]
    for f in sorted(RESULTS_DIR.glob("*/stage5/scale.json")):
        x = _load(f)
        tests[f"stage 5 max-T {x['run']}"] = x["maxT_p"]
    names = list(tests)
    q = false_discovery_control([tests[n] for n in names], method="bh") if names else []
    out = {"n_tests": len(names), "tests": {n: {"p": tests[n], "q": float(qq)} for n, qq in zip(names, q)}}
    (R / "exploratory_bh.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def exploratory_bh_table(run: str) -> str:
    r = _load(RESULTS_DIR / run / "exploratory_bh.json")
    if not r:
        return "*(no exploratory BH results)*\n"
    keep = {k: v for k, v in r["tests"].items() if not k.startswith("H1a")}
    h1a = [v for k, v in r["tests"].items() if k.startswith("H1a")]
    out = [f"BH-FDR over all {r['n_tests']} recorded exploratory p-values in the historical reporting family. The "
           f"{len(h1a)} H1a checks are summarized ({sum(v['q'] < 0.05 for v in h1a)} with q < 0.05; they are per-layer "
           "stage-1a checks, see the stage-1a table).\n", "| test | p | BH q |", "|---|---|---|"]
    for k, v in keep.items():
        name = "Post-edit donor prevalence: probe versus class-only" if k == "C-swap probe > class-only subspace" else k
        out.append(f"| {name} | {p_fmt(v['p'])} | {p_fmt(v['q'])} |")
    out.append("\nRankings in this archive do not recover the registered switch comparison. "
               "The ordinary behavioral and conflict binomial p-values treat reused species as independent.")
    return "\n".join(out) + "\n"


def stage1a_table(run: str) -> str:
    """One row per layer: explained variance, C1, C2 (per zoom stratum), C3, C3b, reading."""
    ch = _load(RESULTS_DIR / run / "stage1a" / "checks.json")
    if not ch:
        return f"*(no stage-1a results for {run})*\n"
    zs = list(ch[0]["C2_within_stratum_order_purity"])
    head = ("| layer | PC1–3 var | C1 purity (null) p | C1 random-init | " + " | ".join(f"C2 {z} (null) p" for z in zs)
            + " | C3 overlay/null p | C3b overlay/within-null p | reading |")
    rows = [head, "|" + "---|" * (7 + len(zs))]
    for c in ch:
        ev = sum(c["explained_variance_ratio"])
        c2 = [f"{v['purity']:.2f} ({v['null']:.2f}) {p_fmt(v['p'])}" for v in c["C2_within_stratum_order_purity"].values()]
        rows.append(
            f"| {c['layer']} | {ev:.1%} | {c['C1_purity']:.2f} ({c['C1_purity_null']:.2f}) {p_fmt(c['C1_p'])} | "
            + (f"{c['C1_purity_random_init']:.2f}" if "C1_purity_random_init" in c else "–") + " | "
            + " | ".join(c2)
            + f" | {c['C3_overlay_length'] / c['C3_null_mean']:.2f} {p_fmt(c['C3_p'])}"
            + f" | {c['C3_overlay_length'] / c['C3b_within_stratum_null_mean']:.2f} {p_fmt(c['C3b_p'])}"
            + f" | {c['reading']} |")
    return "\n".join(rows) + "\n"


def stage1b_tables(run: str) -> str:
    """H1 decision, secondary hypotheses, and TA_within for model, controls and baselines."""
    r = _load(RESULTS_DIR / run / "stage1b" / "results.json")
    if not r:
        return f"*(no stage-1b results for {run})*\n"
    if run == "qwen35_2b":
        from phylogeo.verification.publication import assemble, controls_table, secondary_table
        g, d, _, decisions = assemble()
        return controls_table(g) + "\n\n" + secondary_table(g, d, decisions) + "\n"
    h = r["H1"]
    out = [f"**H1 (primary)** — n = {r['n_species']} species, {r['n_triplets']} sampled triplets, "
           f"{r['n_within']} within-stratum head-noun-disjoint.\n",
           "| quantity | value |", "|---|---|",
           f"| cross-fitted TA_within | {h['crossfit_ta_within']:.3f} |",
           f"| TA_within at modal layer {h['modal_layer']} [95% clade-bootstrap CI] | "
           f"{h['ta_within_modal']:.3f} [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}] |",
           f"| max-T permutation p (all layers) | {p_fmt(h['maxT_p'])} |"]
    for k, v in h["diff_ci95_vs"].items():
        out.append(f"| difference vs {k} [95% CI] | [{v[0]:+.3f}, {v[1]:+.3f}] |")
    for k, v in h["criteria"].items():
        out.append(f"| criterion: {k} | {fmt(v)} |")
    out.append(f"| **H1 supported** | **{fmt(h['supported'])}** |")
    out.append(f"| meaningful effect (≥ 0.40) | {fmt(h['meaningful_effect_ge_0.40'])} |\n")
    out += ["**Secondary hypotheses (Holm-adjusted across S1–S5)**\n", "| id | statistic | value | p | Holm p |",
            "|---|---|---|---|---|"]
    for k, v in r["secondary"].items():
        if k == "S1":
            stat, val = "Mantel Spearman (cosine vs path length) at modal layer", v["mantel_spearman_pathlen_modal"]
        elif k == "S5":
            stat = "pooled within-rank partial Spearman (distance vs TimeTree age, controlling LCA depth)"
            val = v.get("pooled_partial_spearman", v.get("pooled_spearman"))
        else:
            stat, val = f"TA at modal layer (n = {v['n_triplets']} triplets)", v["ta_modal"]
        out.append(f"| {k} | {stat} | {val:.3f} | {p_fmt(v['maxT_p'])} | {p_fmt(v['holm_p'])} |")
    mm0 = r.get("baselines_mantel_pathlen", {})
    if "class_oracle" in mm0 and "log_popularity" in mm0:
        out.append(f"\nContext for S1: the class oracle alone reaches Mantel r = {mm0['class_oracle']:.3f} and log "
                   f"popularity alone {mm0['log_popularity']:.3f}, so S1 does not separate phylogeny from coarse "
                   "classes or popularity.\n")
    ex = r.get("exploratory", {})
    if "posthoc_vs_untrained" in ex:
        ph = ex["posthoc_vs_untrained"]
        ctl = [k for k in next(iter(ph.values())) if k not in ("n_triplets", run, f"{run}_ci95") and not k.startswith("diff_")]
        out += ["\n**Post hoc (final review): model vs the untrained controls on the same subsets** (paired family-block "
                "bootstrap CIs of the difference)\n",
                "| subset | triplets | " + run + " | " + " | ".join(f"{c} (diff CI)" for c in ctl) + " |",
                "|---|---|---|" + "---|" * len(ctl)]
        for name, v in ph.items():
            out.append(f"| {name} | {v['n_triplets']} | {v[run]:.3f} | " + " | ".join(
                f"{v[c]:.3f} ([{v[f'diff_ci95_vs_{c}'][0]:+.3f}, {v[f'diff_ci95_vs_{c}'][1]:+.3f}])" for c in ctl) + " |")
        m = ex["margin_vs_strongest_required"]
        out.append(f"\nMargin over the strongest required control ({m['control']}): {m['margin']:+.3f} "
                   f"[{m['diff_ci95'][0]:+.3f}, {m['diff_ci95'][1]:+.3f}].\n")
    if "per_stratum_ci95" in ex:
        out += ["**Per-stratum TA_within** (layer of H1; family-block bootstrap CI; strata with few decision nodes have "
                "wide CIs)\n", "| stratum | decision nodes | TA_within | 95% CI |", "|---|---|---|---|"]
        for s_, v in sorted(ex["per_stratum_ci95"].items(), key=lambda kv: -kv[1]["n_nodes"]):
            out.append(f"| {s_} | {v['n_nodes']} | {v['ta']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    out += ["\n**TA_within for every representation** (model and controls at their modal layer; chance = 0.333)\n",
            "| representation | TA_within | 95% CI | Mantel r (path length) |", "|---|---|---|---|"]
    mm = r.get("baselines_mantel_pathlen", {})
    for k, v in r["point_ta_within"].items():
        if k not in r["ci95_ta_within"]:  # e.g. glove_known_names_only (no bootstrap)
            out.append(f"| {k} | {v:.3f} | – | – |")
            continue
        lo, hi = r["ci95_ta_within"][k]
        mr = r["models"][k]["mantel_pathlen_by_layer"][r["models"][k]["modal_layer"]] if k in r["models"] else mm.get(k)
        out.append(f"| {k} | {v:.3f} | [{lo:.3f}, {hi:.3f}] | {fmt(mr)} |")
    out += ["\n**Depth profile** (TA over all triplets by deepest rank shared by the three species)\n"]
    levels = list(next(iter(r["depth_profile"].values())))
    out.append("| representation | " + " | ".join(f"{lv} (n={r['depth_counts'][lv]})" for lv in levels) + " |")
    out.append("|---|" + "---|" * len(levels))
    for k, d in r["depth_profile"].items():
        out.append(f"| {k} | " + " | ".join(fmt(d[lv]) for lv in levels) + " |")
    return "\n".join(out) + "\n"


def conflict_table(run: str) -> str:
    if run == "qwen35_2b":
        from phylogeo.verification.publication import conflict_table as verified_table
        return verified_table(_load(RESULTS_DIR / "scientific-verification/outcomes.json")) + "\n"
    c = _load(RESULTS_DIR / run / "conflict" / "conflict.json")
    if not c:
        return f"*(no conflict results for {run})*\n"
    out = ["| representation | layer | chose tree-closer partner | accuracy | binomial p (vs 0.5) | "
           "McNemar vs main model: main-only / control-only, p |", "|---|---|---|---|---|---|"]
    for k, v in c["summary"].items():
        mc = v.get("mcnemar_vs_main")
        mcs = f"{mc['main_only']} / {mc['control_only']}, {p_fmt(mc['p_two_sided'])}" if mc else "–"
        out.append(f"| {k} | {v['layer']} | {v['phylo_choice']}/{v['n']} | {v['accuracy']:.2f} | "
                   f"{p_fmt(v['binom_p_two_sided'])} | {mcs} |")
    out.append("\nThe 36 triplets share anchors and partners (e.g. 'human' is the tree-closer partner several times), so "
               "binomial and McNemar p-values treat dependent items as independent; read them as descriptive.\n")
    rows = [t for t in c["triplets"] if t.get("resolved")]
    main = next(iter(c["summary"]))
    l = c["summary"][main]["layer"]
    out += ["", f"Per triplet ({main}, layer {l}; ✓ = representation closer to the tree-closer partner):", "",
            "| anchor | tree-closer | other | type | " + " | ".join(c["summary"]) + " |",
            "|---|---|---|---|" + "---|" * len(c["summary"])]
    for t in rows:
        marks = []
        for k, v in c["summary"].items():
            marks.append("✓" if t[k][v["layer"]] else "✗")
        out.append(f"| {t['anchor']} | {t['tree_closer']} | {t['other']} | {t['type']} | " + " | ".join(marks) + " |")
    return "\n".join(out) + "\n"


def stage2_table(run: str) -> str:
    """Flat-probe CV correlations (5 seeds, family holdout), delta_rel, NJ RF, geodesic vs angular."""
    r = _load(RESULTS_DIR / run / "stage2" / "results.json")
    if not r:
        return f"*(no stage-2 results for {run})*\n"
    out = [f"Layer {r['layer']} (the stage-1b modal layer).\n",
           "**Flat-subspace probe** (k = 10, PCA-100 fit in-fold, 5-fold family holdout, mean ± sd over seeds)\n",
           "| input / target | held-out-family Pearson r (all pairs) | Spearman | within-stratum Spearman (mean over strata) "
           "| train–test r | train–train r |", "|---|---|---|---|---|---|"]
    for k, v in r["probe"].items():
        if not isinstance(v, dict):
            continue
        ws = v.get("test_test_within_stratum_spearman")
        out.append(f"| {k} | {v['test_test_pearson'][0]:.3f} ± {v['test_test_pearson'][1]:.3f} | "
                   f"{v['test_test_spearman'][0]:.3f} | {fmt(ws[0]) if ws else '–'} | {v['train_test_pearson'][0]:.3f} | "
                   f"{v['train_train_pearson'][0]:.3f} |")
    out.append(f"\nSelectivity (model − shuffled-tree, held-out families): {r['probe']['selectivity_test_test']:.3f}. "
               "± is the SD over probe seeds (folds and initialization), not a CI.\n")
    if "class_only_predictor_pearson_all_pairs" in r["probe"]:
        out.append(f"References without a probe: a predictor that knows only the strata reaches Pearson r = "
                   f"{r['probe']['class_only_predictor_pearson_all_pairs']:.3f} with path length over all pairs, so the "
                   f"stratum labels alone yield a high in-sample association (compare the class_blob input row). The raw "
                   f"angular distances reach a within-stratum Spearman of "
                   f"{r['probe']['raw_angular_within_stratum_spearman_all_species']:.3f} (all species, no probe).\n")
    d = r["delta_rel"]
    out += ["**Gromov δ_rel** (lower = more tree-like; 0 for a tree metric)\n", "| representation | δ_rel |", "|---|---|",
            f"| model | {d['model']:.3f} |",
            f"| spectrum-matched Gaussian (10 draws) | {d['matched_gaussian_mean']:.3f} ± {d['matched_gaussian_sd']:.3f} |",
            f"| class-blob simulation (10 draws) | {d['class_blob_mean']:.3f} ± {d['class_blob_sd']:.3f} |",
            f"| tree oracle (path length) | {d['tree_oracle']:.3f} |"]
    nj = r["nj_rf"]
    out += ["\n**Neighbor-joining tree vs reference** (normalized Robinson–Foulds; 0 = identical)\n",
            "| | RF |", "|---|---|", f"| model | {nj['model']:.3f} |",
            f"| label-permuted (20) | {nj['perm_null_mean']:.3f} ± {nj['perm_null_sd']:.4f} (p = {nj['perm_p']:.3f}) |"]
    if "class_blob_null_mean" in nj:
        out.append(f"| class-blob inputs (20; coarse groups only) | {nj['class_blob_null_mean']:.3f} ± "
                   f"{nj['class_blob_null_sd']:.4f} (p = {nj['class_blob_p']:.3f}) |")
    g = r["geodesic_vs_angular"]
    lay = r["layer"]
    out += ["\n**kNN geodesic vs angular distance** (at the probe layer; gain on the class-blob null for reference)\n",
            "| metric | angular | geodesic | gain | class-blob null gain |", "|---|---|---|---|---|",
            f"| Mantel Spearman vs path length | {g['angular_mantel'][lay]:.3f} | {g['geodesic_mantel'][lay]:.3f} | "
            f"{g['geodesic_mantel'][lay] - g['angular_mantel'][lay]:+.3f} | {g['blob_gain_mantel'][lay]:+.3f} |",
            f"| TA_within | {g['angular_ta'][lay]:.3f} | {g['geodesic_ta'][lay]:.3f} | "
            f"{g['geodesic_ta'][lay] - g['angular_ta'][lay]:+.3f} | {g['blob_gain_ta'][lay]:+.3f} |"]
    if run == "qwen35_2b":
        from phylogeo.verification.statistics import OUT, read_json
        graph = read_json(OUT / "graph.json")["by_layer"]
        out.append(f"\nWith zero-weight edges preserved at layer zero, minimum connected k changes from "
                   f"{graph[0]['archived_k']} to {graph[0]['k']}. Historical plots use the original graph. "
                   "The probe's original target scale included held-out species and the layer was selected on the full panel.")
    return "\n".join(out) + "\n"


def stage4_table(run: str) -> str:
    """Ablation variants, polysemous inclusion, probe dimension/metric, geodesic k sweep."""
    r = _load(RESULTS_DIR / run / "stage4" / "results.json")
    if not r:
        return f"*(no stage-4 results for {run})*\n"
    out = [f"**Single-change ablations** (TA_within on the stage-1b triplets; modal layer {r['modal_layer']})\n",
           "| variant | TA_within (modal layer) | max over layers (layer) | Mantel r (modal) |", "|---|---|---|---|"]
    for k, v in r["variants"].items():
        extra = ""
        if "common_names_same_triplets_modal" in v:
            extra = f" (common names on the same triplets: {v['common_names_same_triplets_modal']:.3f}"
            if "head_disjoint_common_names_modal" in v:
                extra += (f"; head-noun-disjoint subset, n = {v['n_head_disjoint']}: binomials "
                          f"{v['head_disjoint_binomial_modal']:.3f} (max over layers {v['head_disjoint_binomial_max']:.3f}), "
                          f"common names {v['head_disjoint_common_names_modal']:.3f}")
            extra += ")"
        out.append(f"| {k}{extra} | {v['ta_within_modal']:.3f} | {v['ta_within_max']:.3f} ({v['argmax_layer']}) | "
                   f"{v['mantel_pathlen_modal']:.3f} |")
    if "polysemous_included" in r:
        p = r["polysemous_included"]
        out.append(f"\nPolysemous names included (n = {p['n_species']}): TA_within {p['ta_within_modal']:.3f} "
                   f"(primary set {p['ta_within_primary_modal']:.3f}).\n")
    out += ["\n**Probe ablations** (held-out families, mean of 3 seeds)\n",
            "| probe | held-out r (all pairs) | within-stratum Spearman (mean over strata) | pooled within-group Spearman "
            "(Simpson-prone) | train r |", "|---|---|---|---|---|"]
    for k, v in r["probe"].items():
        out.append(f"| {k} | {v['test_test_pearson']:.3f} | {fmt(v.get('test_test_within_stratum_spearman'))} | "
                   f"{v.get('test_test_within_group_spearman', float('nan')):.3f} | {v['train_train_pearson']:.3f} |")
    g = r["geodesic_k_sweep"]
    out += [f"\n**kNN geodesic, k sweep** (smallest connected k = {g['smallest_connected_k']})\n",
            "| k | Mantel r vs path length | TA_within |", "|---|---|---|"]
    for k, v in g["values"].items():
        out.append(f"| {k} | {v['mantel_pathlen']:.3f} | {v['ta_within']:.3f} |")
    return "\n".join(out) + "\n"


def eco_table(run: str) -> str:
    """Addendum A (habitat gradient on held-out species)."""
    if run == "qwen35_2b":
        from phylogeo.verification.publication import habitat_table as verified_table
        return verified_table(_load(RESULTS_DIR / "scientific-verification/outcomes.json")) + "\n"
    r = _load(RESULTS_DIR / run / "eco" / "results.json")
    if not r:
        return f"*(no habitat-addendum results for {run})*\n"
    out = [f"Held-out: {r['n_mammals']} mammals, {r['n_birds']} birds; axis = oriented PC1 of layer {r['layer']} (primary set).\n",
           "| test | statistic | one-sided p | Holm p |", "|---|---|---|---|"]
    for k, v in r["tests"].items():
        if "p" not in v:
            out.append(f"| {k} | not evaluable (n = {v['n']}) | – | – |")
        elif "rho" in v:
            out.append(f"| {k} | Spearman ρ = {v['rho']:.3f} | {p_fmt(v['p'])} | {p_fmt(v['holm_p'])} |")
        else:
            out.append(f"| {k} | median {v['median']:.2f} (n = {v['n']}) vs ground {v['median_ground']:.2f} "
                       f"(n = {v['n_ground']}) | {p_fmt(v['p'])} | {p_fmt(v['holm_p'])} |")
    return "\n".join(out) + "\n"


def stage3_table(run: str) -> str:
    """Behavioral judgments and the causal tests (final run; first run kept as superseded)."""
    if run == "qwen35_2b":
        from phylogeo.verification.publication import causal_table as verified_table
        return verified_table(_load(RESULTS_DIR / "scientific-verification/outcomes.json")) + "\n"
    r = _load(RESULTS_DIR / run / "stage3" / "results.json")
    if not r:
        return f"*(no stage-3 results for {run})*\n"
    b, c, s = r["behavior_conflict"], r["behavior_species_triplets"], r["class_accuracy"]
    L = r["layer"]
    out = ["**Behavior** (answer-letter log-odds, averaged over both answer orders; binomial test vs 0.5, two-sided)\n",
           "| test | n | accuracy | binomial p |", "|---|---|---|---|",
           f"| conflict triplets: chose the tree-closer animal | {b['n']} | {b['accuracy']:.3f} | {p_fmt(b['binom_p'])} |",
           f"| within-stratum species triplets: chose the tree-closer animal | {c['n']} | "
           f"{c['behavioral_accuracy']:.3f} | {p_fmt(c['binom_p'])} |",
           f"| class word correct (argmax over candidate class words) | {s['n']} | {s['accuracy']:.3f} | – |",
           f"\nOn the same species triplets, the layer-{L} representation is closer to the tree-closer animal in "
           f"{c['representation_pairwise_accuracy_same_triplets']:.3f} of cases; behavior and representation agree on "
           f"{c['agreement_behavior_vs_representation']:.3f}.\n",
           "Class-word accuracy by stratum: " + ", ".join(f"{k} {v:.2f}" for k, v in s["by_stratum"].items()) + ".\n"]
    for ka, ks in (("C_abl", "C_swap"), ("C_abl_v1", "C_swap_v1")):
        if ka not in r:
            continue
        a, w = r[ka], r[ks]
        # Label by content, not by key (final review M5): only the deviation-19 design counts as final.
        final = ka == "C_abl" and "causal_holm" in r and a.get("n_random_subspaces", 0) >= 99
        tag = "final run (deviations 19-20)" if final else "first run (superseded: p floor above the Holm threshold)"
        holm_p = r["causal_holm"] if final else {}
        na, ns = a.get("n_random_subspaces", len(a["random_mean_drops"])), len(w["switch_rate_random"])
        out += [f"\n**Causal tests, {tag}** (layer {L}; k = 10 probe subspace fit on half of the families; "
                f"{a['n_test_species']} held-out species; captured variance {a['probe_var_captured']:.3f})\n",
                "| hypothesis | probe subspace | random subspaces: mean [max] | rank p | Holm p | secondary |",
                "|---|---|---|---|---|---|",
                f"| C-abl: drop in log P(correct class) | {a['mean_drop_logp_probe']:.3f} | "
                f"{np.mean(a['random_mean_drops']):.3f} [{np.max(a['random_mean_drops']):.3f}] (n = {na}) | "
                f"{p_fmt(a['p_rank_vs_random'])} | {p_fmt(holm_p['C_abl']) if holm_p else '–'} | "
                f"Wilcoxon over species p = {p_fmt(a['wilcoxon_p_probe_gt_random'])} |",
                f"| C-swap: answer switches to the source's class | {w['switch_rate_probe']:.3f} | "
                f"{np.mean(w['switch_rate_random']):.3f} [{np.max(w['switch_rate_random']):.3f}] (n = {ns}) | "
                f"{p_fmt(w['p_rank_vs_random'])} | {p_fmt(holm_p['C_swap']) if holm_p else '–'} | "
                f"no edit: {w['switch_rate_no_edit']:.3f} ({w['n_pairs']} pairs) |"]
        if "mean_base_logp_correct" in a:
            out.append(f"\nMean log P(correct class) without editing: {a['mean_base_logp_correct']:.3f}.")
        if final:
            out.append(f"\nOff-target next-token loss on neutral sentences: no edit {a['neutral_loss_base']:.4f}, probe "
                       f"subspace ablated {a['neutral_loss_probe_ablated']:.4f}, random ablated "
                       f"{a['neutral_loss_random_ablated']:.4f}.\n")
        else:
            out.append("\nOff-target losses of this run are not reported: its no-edit baseline used a different "
                       "sentence set and weighting (deviation 19).\n")
        if final and "class_control" in r:
            cc = r["class_control"]
            out += ["**Class-only subspace control** (deviation 20; the same probe trained on the class-oracle distance)\n",
                    "| quantity | value |", "|---|---|",
                    f"| captured variance, class-only subspace | {cc['class_var_captured']:.3f} |",
                    f"| mean drop in log P(correct class), class-only subspace | {cc['mean_drop_logp_class']:.3f} |",
                    f"| phylogeny probe > class-only, ablation (one-sided Wilcoxon over species) p | "
                    f"{p_fmt(cc['wilcoxon_p_probe_gt_class'])} |",
                    f"| switch rate, class-only subspace | {cc['switch_rate_class']:.3f} |",
                    f"| phylogeny probe > class-only, interchange (exact McNemar; {cc['mcnemar_probe_only']} vs "
                    f"{cc['mcnemar_class_only']} discordant pairs) p | {p_fmt(cc['mcnemar_p_probe_gt_class'])} |",
                    f"| **phylogeny-specific causal role** (both p < 0.05) | **{fmt(cc['phylogeny_specific'])}** |"]
    return "\n".join(out) + "\n"


def stage5_table() -> str:
    """Scale and training-dynamics runs (one row per run)."""
    from phylogeo.verification.publication import scale_table
    return scale_table()


def stage6_table(run: str) -> str:
    """Spread of the headline numbers under each perturbation."""
    r = _load(RESULTS_DIR / run / "stage6" / "results.json")
    if not r:
        return f"*(no stage-6 results for {run})*\n"

    def ms(s):
        return f"{s['mean']:.3f} ± {s['sd']:.3f} [{s['p2.5']:.3f}, {s['p97.5']:.3f}]"

    out = [f"Layer {r['layer']}; main-run TA_within {r['main_run'][run]:.3f}. Mean ± SD [2.5th, 97.5th percentile]"
           " over repetitions; margin = model minus the best of the untrained and nuisance controls in the same "
           "repetition.\n", "| perturbation | repetitions | TA_within | min | margin over best control (min) |",
           "|---|---|---|---|---|"]
    for key, label in (("triplet_seed", "triplet sampling seed"), ("species_subsample", "80% species subsample")):
        b = r[key]
        out.append(f"| {label} | {len(b[run]['values'])} | {ms(b[run])} | {b[run]['min']:.3f} | "
                   f"{b['margin_vs_best_control']['mean']:.3f} ({b['margin_vs_best_control']['min']:.3f}) |")
    for key, label in (("half", "template halves (4 of 8)"), ("leave_one_out", "leave one template out"),
                       ("single", "single template")):
        t = r["templates"][key]
        out.append(f"| {label} | {len(t['ta_within']['values'])} | {ms(t['ta_within'])} | {t['ta_within']['min']:.3f}"
                   f" | – (best layer: {', '.join(map(str, t['argmax_layer']))}) |")
    cf = r["crossfit_seed"]
    out.append(f"| cross-fit family-split seed | {len(cf['modal_layer'])} | {ms(cf['crossfit_ta_within'])} | "
               f"{cf['crossfit_ta_within']['min']:.3f} | – (modal layer: {', '.join(map(str, cf['modal_layer']))}) |")
    k = r["k_per_node"]
    out += ["\n**Triplets per decision node K** (one draw each)\n",
            "| K | within triplets | " + " | ".join(r["representations"]) + " |",
            "|---|---|" + "---|" * len(r["representations"])]
    out += [f"| {K} | {v['n_within']} | " + " | ".join(f"{v[x]:.3f}" for x in r["representations"]) + " |"
            for K, v in k.items()]
    p = r["probe_seed"]
    out.append(f"\n**Flat probe, {len(p['test_test_pearson']['values'])} fresh seeds** (folds and init): held-out "
               f"Pearson r {ms(p['test_test_pearson'])}; within-stratum Spearman (mean over strata) "
               + (ms(p["test_test_within_stratum_spearman"]) if "test_test_within_stratum_spearman" in p else "–")
               + f"; pooled within-group Spearman (Simpson-prone) {ms(p['test_test_within_group_spearman'])}.\n")
    sp = r["species_subsample"]["mantel_pathlen_model"]
    out.append(f"Mantel r vs path length under 80% species subsamples: {ms(sp)}; under template halves: "
               f"{ms(r['templates']['half']['mantel_pathlen'])}.\n")
    return "\n".join(out) + "\n"


def key_results_md(run: str = "qwen35_2b") -> str:
    """The public hypothesis decisions are generated from the independent frozen-data replay."""
    from phylogeo.verification.publication import headline
    from phylogeo.verification.statistics import MAIN
    return headline() if run == MAIN else "Exploratory pilot. See its archived run tables."

def inject_block(path: Path, text: str, tag: str = "KEY-RESULTS") -> bool:
    """Replace the text between <!-- {tag}:START --> and <!-- {tag}:END --> in ``path``."""
    if not path.exists():
        return False
    s = path.read_text(encoding="utf-8")
    a, z = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if a not in s or z not in s:
        return False
    path.write_text(s[: s.index(a) + len(a)] + "\n" + text + s[s.index(z):], encoding="utf-8")
    return True


def probe_bar_png(run: str) -> None:
    """Two panels: held-out probe correlation over all pairs, and within strata (mean of per-stratum
    Spearman), for the model and every control. Error bars = SD over probe seeds (not CIs)."""
    r = _load(RESULTS_DIR / run / "stage2" / "results.json")
    if not r:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k, v in r["probe"].items() if isinstance(v, dict)]
    panels = [("test_test_pearson", "all held-out pairs: Pearson r")]
    if all("test_test_within_stratum_spearman" in r["probe"][k] for k in keys):
        panels.append(("test_test_within_stratum_spearman", "within strata: mean Spearman"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        m = [r["probe"][k][key][0] for k in keys]
        sd = [r["probe"][k][key][1] for k in keys]
        ax.bar(keys, m, yerr=sd, color=["#2a78d6" if k == "model" else "#b8b7b2" for k in keys], capsize=4,
               edgecolor="white", linewidth=1)
        ax.axhline(0, color="0.3", lw=0.8)
        if key == "test_test_pearson" and "class_only_predictor_pearson_all_pairs" in r["probe"]:
            v = r["probe"]["class_only_predictor_pearson_all_pairs"]
            ax.axhline(v, ls=":", color="0.4", lw=1)
            ax.text(len(keys) - 0.5, v, "class-only predictor ", ha="right", va="bottom", fontsize=7, color="0.3")
        if key != "test_test_pearson" and "raw_angular_within_stratum_spearman_all_species" in r["probe"]:
            v = r["probe"]["raw_angular_within_stratum_spearman_all_species"]
            ax.axhline(v, ls=":", color="0.4", lw=1)
            ax.text(len(keys) - 0.5, v, "raw distances, no probe ", ha="right", va="bottom", fontsize=7, color="0.3")
        for i_, v in enumerate(m):
            ax.text(i_, v + 0.02 if v >= 0 else v - 0.06, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    axes[0][0].set_ylabel("held-out families (5-fold)")
    fig.suptitle(f"Flat-subspace probe (k = 10), layer {r['layer']}: predicted vs tree distance; bars = mean, "
                 "error = SD over seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / run / "stage2" / "probe_heldout_r.png", dpi=150)
    plt.close(fig)


def power_table() -> str:
    s = _load(RESULTS_DIR / "power" / "summary.json")
    if not s:
        return "*(no power results)*\n"
    from phylogeo.verification.statistics import OUT, read_json
    replay = read_json(OUT / "outcomes.json")["exploratory"]
    out = [f"Single-layer permutation rejection frequencies, {replay['power_reps']} simulations per setting. "
           "Intervals describe Monte Carlo uncertainty. Power of the joint H1 rule is unknown.\n",
           "| signal | mean TA_within | SD | rejection fraction | exact 95% CI |", "|---|---|---|---|---|"]
    for r in replay["power"]:
        lo, hi = r["binomial_ci"]
        out.append(f"| {r['signal']} | {r['ta_within_mean']:.3f} | {r['ta_within_sd']:.3f} | "
                   f"{r['power']:.2f} | [{lo:.3f}, {hi:.3f}] |")
    return "\n".join(out) + "\n"




def write_results_md(runs: list[str], main: str = "qwen35_2b") -> Path:
    """Aggregate all generated tables into results/RESULTS.md."""
    parts = ["# Results\n", key_results_md(main), "\nAll stage-specific exploratory tables below retain their original scoring and test families unless explicitly identified as corrected.\n",
             "## Power simulation\n", power_table()]
    for run in runs:
        parts += [f"\n## Run `{run}`\n", "### Stage 1a — top-3 PCA checks per layer\n", stage1a_table(run),
                  "\n### Stage 1b — triplet accuracy and secondary hypotheses\n", stage1b_tables(run),
                  "\n### Conflict triplets (phylogeny vs folk/lexical)\n", conflict_table(run),
                  "\n### Stage 2 — geometry and flat-subspace probe\n", stage2_table(run),
                  "\n### Stage 3 — behavior and causal tests\n", stage3_table(run),
                  "\n### Stage 4 — ablations\n", stage4_table(run),
                  "\n### Addendum A — habitat gradient (held-out species)\n", eco_table(run),
                  "\n### Stage 6 — sensitivity\n", stage6_table(run)]
    parts += ["\n## Stage 5 — more models (scale and training dynamics)\n", stage5_table()]
    exploratory_bh(main)  # registered BH-FDR family (final review M7); read by key_results_md and the paper
    parts += ["\n## Exploratory tests: Historical BH adjustment over the recorded tests\n", exploratory_bh_table(main)]
    from phylogeo.verification.publication import write_review
    write_review()
    path = RESULTS_DIR / "RESULTS.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    from phylogeo.config import REPO_ROOT

    for doc in (REPO_ROOT / "README.md", REPO_ROOT / "reports" / "summary.md"):  # generated headline blocks
        inject_block(doc, key_results_md(main))
    for run in runs:
        probe_bar_png(run)
    for run in runs:  # per-run summary next to the run's results (linked from experiment notes)
        (RESULTS_DIR / run / "SUMMARY.md").write_text(
            "\n".join([f"# Run `{run}` (auto-generated; do not edit)\n", "## Stage 1a\n", stage1a_table(run),
                       "\n## Stage 1b\n", stage1b_tables(run), "\n## Conflict triplets\n", conflict_table(run),
                       "\n## Stage 2\n", stage2_table(run), "\n## Stage 3\n", stage3_table(run),
                       "\n## Stage 4\n", stage4_table(run), "\n## Addendum A (habitat gradient)\n", eco_table(run),
                       "\n## Stage 6\n", stage6_table(run)]),
            encoding="utf-8")
    return path
