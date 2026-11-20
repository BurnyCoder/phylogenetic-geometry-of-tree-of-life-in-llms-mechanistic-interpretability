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
    return "<0.001" if p < 0.001 else f"{p:.3f}"


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
    c = _load(RESULTS_DIR / run / "conflict" / "conflict.json")
    if not c:
        return f"*(no conflict results for {run})*\n"
    out = ["| representation | layer | chose tree-closer partner | accuracy | binomial p (vs 0.5) |",
           "|---|---|---|---|---|"]
    for k, v in c["summary"].items():
        out.append(f"| {k} | {v['layer']} | {v['phylo_choice']}/{v['n']} | {v['accuracy']:.2f} | "
                   f"{p_fmt(v['binom_p_two_sided'])} |")
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
           "| input / target | held-out-family Pearson r | Spearman | train–test r | train–train r |",
           "|---|---|---|---|---|"]
    for k, v in r["probe"].items():
        if k == "selectivity_test_test":
            continue
        out.append(f"| {k} | {v['test_test_pearson'][0]:.3f} ± {v['test_test_pearson'][1]:.3f} | "
                   f"{v['test_test_spearman'][0]:.3f} | {v['train_test_pearson'][0]:.3f} | {v['train_train_pearson'][0]:.3f} |")
    out.append(f"\nSelectivity (model − shuffled-tree, held-out families): **{r['probe']['selectivity_test_test']:.3f}**\n")
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
    g = r["geodesic_vs_angular"]
    lay = r["layer"]
    out += ["\n**kNN geodesic vs angular distance** (at the probe layer; gain on the class-blob null for reference)\n",
            "| metric | angular | geodesic | gain | class-blob null gain |", "|---|---|---|---|---|",
            f"| Mantel Spearman vs path length | {g['angular_mantel'][lay]:.3f} | {g['geodesic_mantel'][lay]:.3f} | "
            f"{g['geodesic_mantel'][lay] - g['angular_mantel'][lay]:+.3f} | {g['blob_gain_mantel'][lay]:+.3f} |",
            f"| TA_within | {g['angular_ta'][lay]:.3f} | {g['geodesic_ta'][lay]:.3f} | "
            f"{g['geodesic_ta'][lay] - g['angular_ta'][lay]:+.3f} | {g['blob_gain_ta'][lay]:+.3f} |"]
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
            extra = f" (common names on the same triplets: {v['common_names_same_triplets_modal']:.3f})"
        out.append(f"| {k}{extra} | {v['ta_within_modal']:.3f} | {v['ta_within_max']:.3f} ({v['argmax_layer']}) | "
                   f"{v['mantel_pathlen_modal']:.3f} |")
    if "polysemous_included" in r:
        p = r["polysemous_included"]
        out.append(f"\nPolysemous names included (n = {p['n_species']}): TA_within {p['ta_within_modal']:.3f} "
                   f"(primary set {p['ta_within_primary_modal']:.3f}).\n")
    out += ["\n**Probe ablations** (held-out families, mean of 3 seeds)\n",
            "| probe | held-out r | within-group held-out Spearman | train r |", "|---|---|---|---|"]
    for k, v in r["probe"].items():
        out.append(f"| {k} | {v['test_test_pearson']:.3f} | {v.get('test_test_within_group_spearman', float('nan')):.3f} | "
                   f"{v['train_train_pearson']:.3f} |")
    g = r["geodesic_k_sweep"]
    out += [f"\n**kNN geodesic, k sweep** (smallest connected k = {g['smallest_connected_k']})\n",
            "| k | Mantel r vs path length | TA_within |", "|---|---|---|"]
    for k, v in g["values"].items():
        out.append(f"| {k} | {v['mantel_pathlen']:.3f} | {v['ta_within']:.3f} |")
    return "\n".join(out) + "\n"


def eco_table(run: str) -> str:
    """Addendum A (habitat gradient on held-out species)."""
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


def probe_bar_png(run: str) -> None:
    """Bar chart of held-out-family probe correlation: model vs controls (error bars = seed sd)."""
    r = _load(RESULTS_DIR / run / "stage2" / "results.json")
    if not r:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in r["probe"] if k != "selectivity_test_test"]
    m = [r["probe"][k]["test_test_pearson"][0] for k in keys]
    sd = [r["probe"][k]["test_test_pearson"][1] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(keys, m, yerr=sd, color=["#2a78d6" if k == "model" else "#b8b7b2" for k in keys], capsize=4)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("Pearson r, held-out families")
    ax.set_title(f"Flat-subspace probe (k=10), layer {r['layer']}: predicted vs tree distance", fontsize=9)
    for i, v in enumerate(m):
        ax.text(i, v + 0.02 if v >= 0 else v - 0.06, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / run / "stage2" / "probe_heldout_r.png", dpi=150)
    plt.close(fig)


def power_table() -> str:
    s = _load(RESULTS_DIR / "power" / "summary.json")
    if not s:
        return "*(no power results)*\n"
    out = [f"Minimum detectable TA_within at 80% power: **{fmt(s['min_detectable_ta_within_80pct_power'])}**\n",
           "| signal | mean TA_within | sd | power |", "|---|---|---|---|"]
    for r in s["curve"]:
        out.append(f"| {r['signal']} | {r['ta_within_mean']:.3f} | {r['ta_within_sd']:.3f} | {r['power']:.2f} |")
    return "\n".join(out) + "\n"


def write_results_md(runs: list[str]) -> Path:
    """Aggregate all generated tables into results/RESULTS.md."""
    parts = ["# Results (auto-generated by `python -m phylogeo.run --phases report`; do not edit)\n",
             "## Power simulation\n", power_table()]
    for run in runs:
        parts += [f"\n## Run `{run}`\n", "### Stage 1a — top-3 PCA checks per layer\n", stage1a_table(run),
                  "\n### Stage 1b — triplet accuracy and secondary hypotheses\n", stage1b_tables(run),
                  "\n### Conflict triplets (phylogeny vs folk/lexical)\n", conflict_table(run),
                  "\n### Stage 2 — geometry and flat-subspace probe\n", stage2_table(run),
                  "\n### Stage 4 — ablations\n", stage4_table(run),
                  "\n### Addendum A — habitat gradient (held-out species)\n", eco_table(run)]
    path = RESULTS_DIR / "RESULTS.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    for run in runs:
        probe_bar_png(run)
    for run in runs:  # per-run summary next to the run's results (linked from experiment notes)
        (RESULTS_DIR / run / "SUMMARY.md").write_text(
            "\n".join([f"# Run `{run}` (auto-generated; do not edit)\n", "## Stage 1a\n", stage1a_table(run),
                       "\n## Stage 1b\n", stage1b_tables(run), "\n## Conflict triplets\n", conflict_table(run),
                       "\n## Stage 2\n", stage2_table(run), "\n## Stage 4\n", stage4_table(run),
                       "\n## Addendum A (habitat gradient)\n", eco_table(run)]),
            encoding="utf-8")
    return path
