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
                  "\n### Conflict triplets (phylogeny vs folk/lexical)\n", conflict_table(run)]
    path = RESULTS_DIR / "RESULTS.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
