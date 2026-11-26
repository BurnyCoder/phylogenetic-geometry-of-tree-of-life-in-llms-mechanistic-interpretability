"""Labelled interactive figures (Plotly) for the site, exported as PNG for the README.

Global context: every figure is defined once here and read only from ``results/`` (JSON and
the compact ``results/<run>/viz/`` files written by the ``viz_data`` phase). ``build_figures``
returns a list of figure records; the site embeds them as interactive Plotly HTML and
``export_pngs`` writes static copies (via kaleido) that the README shows near the top.
Each record carries a *kind* (ground truth, positive, negative, mixed, example) and a verdict
computed from the results, so the label on a chart can never disagree with the numbers.

Style follows the dataviz skill's validated palette: four categorical hues in fixed order
(blue = this model, gray = untrained networks, orange = name statistics / controls,
aqua = ground-truth references), a single-hue blue ramp for ordered values, and status colours
only for verdict badges, always with a text label.
Plotly API: https://plotly.com/python/reference/ ; static export: https://plotly.com/python/static-image-export/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phylogeo.config import RESULTS_DIR

log = logging.getLogger("phylogeo.figures")

# --- palette (dataviz skill, references/palette.md; validated with validate_palette.js --pairs all) ---
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SURFACE, INK, INK2, MUTED, GRID, AXIS, LIGHTGRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#d6d5d0"
RAMP = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]  # blue 700 -> 100 (dark = close / same group)
LEVEL_COLORS = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f0efec"]  # genus ... different phylum
STATUS = {"positive": "#0ca30c", "negative": "#d03b3b", "mixed": "#fab219", "ground truth": BLUE, "example": GRAY}
ICON = {"positive": "✓", "negative": "✗", "mixed": "◐", "ground truth": "◆", "example": "●"}

DISPLAY = {"qwen35_2b": "Qwen3.5-2B (this model)", "qwen35_2b_random": "Qwen3.5-2B, random weights",
           "pythia410m_step0": "Pythia-410m, step 0 (untrained)", "pythia410m": "Pythia-410m (trained)",
           "char3": "character 3-grams", "glove": "GloVe co-occurrence", "token_count": "name token count",
           "log_popularity": "popularity (log sitelinks)", "input_embedding": "input embedding (layer 0)",
           "class_oracle": "class labels only (oracle)", "rank_oracle": "Linnaean ranks (oracle)",
           "tree_oracle": "exact tree (ground truth)", "tree": "reference tree (ground truth)"}
CATEGORY = {"qwen35_2b": ("this model", BLUE, "circle"), "pythia410m": ("other trained LM", BLUE, "circle-open"),
            "qwen35_2b_random": ("untrained network", GRAY, "circle"), "pythia410m_step0": ("untrained network", GRAY, "diamond"),
            "char3": ("name statistic", ORANGE, "circle"), "glove": ("name statistic", ORANGE, "diamond"),
            "token_count": ("name statistic", ORANGE, "square"), "log_popularity": ("name statistic", ORANGE, "x"),
            "input_embedding": ("name statistic", ORANGE, "triangle-up"), "class_oracle": ("reference oracle", AQUA, "square"),
            "rank_oracle": ("reference oracle", AQUA, "diamond")}
RANK_LABELS = {"none": "different phyla", "phylum": "same phylum,<br>different classes",
               "class": "same class,<br>different orders", "order": "same order,<br>different families",
               "family": "same family,<br>different genera", "genus": "same genus"}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _p(p: float) -> str:
    from phylogeo.report import p_fmt

    return p_fmt(p)


def _style(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    """Shared layout: light chart surface, recessive grid and axes, readable hover labels."""
    fig.update_layout(template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
                      font={"family": "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif", "size": 13,
                            "color": INK},
                      margin={"l": 70, "r": 30, "t": 60, "b": 60}, showlegend=legend,
                      hoverlabel={"bgcolor": "white", "font_size": 12, "bordercolor": AXIS},
                      legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 12}})
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    for a in fig.layout.annotations:  # subplot titles only (paper-referenced); keep custom annotation fonts
        if a.xref == "paper" and a.yref == "paper":
            a.font = {"size": 14, "color": INK2}
    return fig


def _legend_below(fig: go.Figure, bottom: int = 130, **extra) -> go.Figure:
    """Horizontal legend at the bottom edge of the figure (container coordinates), below the x-axis title."""
    return fig.update_layout(legend={"orientation": "h", "yref": "container", "y": 0.01, "yanchor": "bottom", "x": 0,
                                     **extra}, margin={"b": bottom})


def _chance(fig, axis="y", row=None, col=None, text="chance 1/3"):
    """Dashed reference line at the 1/3 chance level of rooted-triplet accuracy."""
    kw = {"line_dash": "dash", "line_color": MUTED, "line_width": 1, "annotation_text": text,
          "annotation_font_color": INK2, "annotation_font_size": 11}
    if row is not None:
        kw.update(row=row, col=col)
    (fig.add_hline if axis == "y" else fig.add_vline)(1 / 3, **kw)


def rec(key, title, kind, verdict, caption, fig, png_size=(1100, 600)) -> dict:
    """One figure record: ``kind`` picks the badge colour, ``verdict`` is the badge text."""
    return {"key": key, "title": title, "kind": kind, "verdict": verdict, "caption": caption, "fig": fig,
            "png_size": png_size}


# ======================================================================================
# Ground truth and expected shapes
# ======================================================================================

def fig_taxonomy(df) -> dict:
    """Sunburst of the reference taxonomy (phylum -> class -> order -> family -> species)."""
    from phylogeo.viz.pca import STRATUM_COLORS

    d = df.copy()
    for c in ["phylum", "class", "order", "family"]:
        d[c] = d[c].replace("", np.nan).fillna("(unranked)")
    d["group"] = [STRATUM_COLORS.get(s, GRAY) for s in d["stratum"]]
    d["n"] = 1
    fig = px.sunburst(d, path=[px.Constant("Animalia"), "phylum", "class", "order", "family", "common_name"],
                      values="n", color="group", color_discrete_map={c: c for c in d["group"].unique()} | {"(?)": GRID},
                      maxdepth=4)
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value} species<br>%{id}<extra></extra>",
                      insidetextorientation="radial", marker={"line": {"color": SURFACE, "width": 1.5}})
    _style(fig, 720, legend=False).update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return rec("taxonomy", "Ground truth: the 400 species on the tree of life", "ground truth", "reference",
               "Rings from the centre: phylum, class, order (click a slice to drill down to families and species). "
               "Colour = super-group: blue vertebrates, orange arthropods, aqua other invertebrates, gray other phyla. "
               "Taxonomy from Wikidata/NCBI; the reference topology used in all tests is the Open Tree of Life.",
               fig, (900, 900))


def _strata_ticks(names_ord, strata_ord, min_run=8):
    """Tick positions at the middle of each contiguous stratum block (blocks of >= min_run species)."""
    vals, text, start = [], [], 0
    for i in range(1, len(strata_ord) + 1):
        if i == len(strata_ord) or strata_ord[i] != strata_ord[start]:
            if i - start >= min_run:
                vals.append(names_ord[(start + i - 1) // 2])
                text.append(strata_ord[start])
            start = i
    return vals, text


def fig_distance_heatmaps(run, df) -> dict | None:
    """2x2 heatmaps of distance percentiles in tree order: ground truth, model, untrained, class-only."""
    f = RESULTS_DIR / run / "viz" / "distances.npz"
    if not f.exists():
        return None
    z = np.load(f)
    order = z["order"]
    names = np.array(df["common_name"])[order].tolist()
    strata = np.array(df["stratum"])[order].tolist()
    L = _load(RESULTS_DIR / run / "viz" / "neighbors.json")["layer"]
    panels = [("tree", "Ground truth: reference tree path length"), ("model", f"Qwen3.5-2B, layer {L} (cosine)"),
              ("untrained", f"Untrained Qwen3.5-2B, layer {L}"), ("class_only", "Class labels only (folk-category oracle)")]
    fig = make_subplots(2, 2, subplot_titles=[t for _, t in panels], horizontal_spacing=0.09, vertical_spacing=0.09)
    scale = [[i / (len(RAMP) - 1), c] for i, c in enumerate(RAMP)]
    for i, (k, t) in enumerate(panels):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_heatmap(z=z[k], x=names, y=names, colorscale=scale, zmin=0, zmax=100, row=r, col=c,
                        showscale=(i == 1), colorbar={"title": {"text": "distance<br>percentile", "side": "top"},
                                                      "tickvals": [0, 50, 100],
                                                      "ticktext": ["0 (closest)", "50", "100 (farthest)"], "len": 0.45,
                                                      "y": 0.78},
                        hovertemplate=f"<b>%{{y}}</b> ↔ <b>%{{x}}</b><br>{t}<br>percentile %{{z}}<extra></extra>")
    tv, tt = _strata_ticks(names, strata)
    fig.update_xaxes(tickvals=tv, ticktext=tt, tickangle=-60, tickfont={"size": 9}, matches="x", showgrid=False)
    fig.update_yaxes(tickvals=tv, ticktext=tt, tickfont={"size": 9}, autorange="reversed", matches="y", showgrid=False)
    fig.update_xaxes(showticklabels=False, row=1)  # the bottom row carries the shared stratum labels
    _style(fig, 1060, legend=False).update_layout(margin={"l": 110, "r": 30, "t": 60, "b": 110})
    return rec("heatmaps", "Expected vs observed: who is close to whom", "ground truth", "ground truth vs observed",
               "Each pixel is a species pair; rows and columns follow the reference tree (depth-first), so clades are "
               "square blocks along the diagonal. Dark = close. Distances are converted to percentiles so the panels "
               "share one colour scale. The model reproduces the coarse blocks (like the class-only oracle) and some of "
               "the finer structure inside them, which the untrained network lacks. Interactive: drag to zoom (all "
               "panels zoom together), hover for the species pair.", fig, (1100, 1080))


def fig_expected_pca(run) -> dict | None:
    """2-D PCA: a true tree (Brownian motion on the reference tree) vs the model and the untrained network."""
    from phylogeo.viz.pca import STRATUM_STYLE

    d = _load(RESULTS_DIR / run / "viz" / "pca2d.json")
    if not d:
        return None
    keys = list(d["panels"])
    titles = {"expected": "Expected if the model stored the tree<br>(Brownian motion on the reference tree)"}
    for k in keys[1:]:
        r, lay = k.split("@")
        titles[k] = (f"Untrained Qwen3.5-2B, layer {lay}" if "random" in r else f"Qwen3.5-2B, layer {lay}")
    fig = make_subplots(2, 2, subplot_titles=[titles[k] for k in keys[:4]], horizontal_spacing=0.08,
                        vertical_spacing=0.13)
    strata, names = np.array(d["strata"]), np.array(d["names"])
    for i, k in enumerate(keys[:4]):
        xy, evr = np.array(d["panels"][k]["xy"]), d["panels"][k]["evr"]
        r, c = i // 2 + 1, i % 2 + 1
        for s, (color, _, sym) in STRATUM_STYLE.items():
            m = strata == s
            if not m.any():
                continue
            fig.add_scatter(x=xy[m, 0], y=xy[m, 1], mode="markers", name=s, legendgroup=s, showlegend=(i == 0),
                            marker={"color": color, "symbol": sym, "size": 7, "line": {"color": SURFACE, "width": 1},
                                    "opacity": 0.9},
                            text=names[m], hovertemplate="<b>%{text}</b><br>" + s + "<extra></extra>", row=r, col=c)
        fig.update_xaxes(title_text=f"PC1 ({100 * evr[0]:.1f}%)", row=r, col=c, showticklabels=False)
        fig.update_yaxes(title_text=f"PC2 ({100 * evr[1]:.1f}%)", row=r, col=c, showticklabels=False)
    _style(fig, 900).update_layout(legend={"title": {"text": "stratum (colour = super-group)"}, "itemsizing": "constant"})
    return rec("pca_expected", "Expected vs observed shape in the top two principal components", "mixed",
               "class arms, no visible nested tree",
               "Top left is the shape a Brownian reference draw would give (the stage-1a synthetic positive control): nested "
               "clusters inside clusters. The model separates coarse groups (class arms) and, in late layers, orders "
               "animals along an aquatic-to-terrestrial axis; the nesting is not visible in two components, which hold "
               "a small share of the variance. The untrained network's top components are dominated by a few outlier names. Click "
               "legend entries to isolate a stratum; hover for names.",
               fig, (1100, 950))


# ======================================================================================
# Primary test (positive)
# ======================================================================================

def fig_h1_forest(run) -> dict | None:
    s = _load(RESULTS_DIR / run / "stage1b" / "results.json")
    if not s:
        return None
    from phylogeo.verification.publication import assemble
    g, _, _, decision = assemble()
    family = g["family"]
    pt = {k: v["ta"] for k, v in family.items()}
    ci = {k: v["ci"] for k, v in family.items()}
    reps = [r for r in CATEGORY if r in pt and r in ci]
    reps.sort(key=lambda r: pt[r])
    fig = make_subplots(1, 2, column_widths=[0.56, 0.44], horizontal_spacing=0.2,
                        subplot_titles=["A. Within-group triplet accuracy (95% CI)",
                                        "B. Paired difference: this model − required control"])
    seen = set()
    for r in reps:
        cat, color, sym = CATEGORY[r]
        lo, hi = ci[r]
        fig.add_scatter(x=[pt[r]], y=[DISPLAY[r]], mode="markers", name=cat, legendgroup=cat, showlegend=cat not in seen,
                        marker={"color": color, "symbol": sym, "size": 12 if r == run else 9,
                                "line": {"color": color, "width": 2}},
                        error_x={"type": "data", "symmetric": False, "array": [hi - pt[r]], "arrayminus": [pt[r] - lo],
                                 "color": color, "thickness": 2, "width": 4},
                        hovertemplate=f"<b>{DISPLAY[r]}</b><br>TA_within %{{x:.3f}}<br>95% CI {lo:.3f}–{hi:.3f}"
                                      f"<extra>{cat}</extra>", row=1, col=1)
        seen.add(cat)
    _chance(fig, "x", 1, 1)
    fig.add_vline(0.40, line_dash="dot", line_color=INK2, line_width=1, row=1, col=1,
                  annotation_text="smallest effect<br>of interest 0.40", annotation_position="bottom right",
                  annotation_font_size=11, annotation_font_color=INK2)
    fig.update_xaxes(range=[0.2, 0.72], title_text="TA_within (chance = 1/3)", row=1, col=1)
    req = [k for k in family if k != run]
    for r in req:
        lo, hi = family[r]["difference_ci"]
        d = pt[run] - pt[r]
        fig.add_scatter(x=[d], y=[DISPLAY[r]], mode="markers", showlegend=False,
                        marker={"color": BLUE, "size": 10},
                        error_x={"type": "data", "symmetric": False, "array": [hi - d], "arrayminus": [d - lo],
                                 "color": BLUE, "thickness": 2, "width": 4},
                        hovertemplate=f"<b>vs {DISPLAY[r]}</b><br>difference %{{x:+.3f}}<br>95% CI {lo:+.3f} to "
                                      f"{hi:+.3f}<extra></extra>", row=1, col=2)
    fig.add_vline(0, line_color=INK2, line_width=1, row=1, col=2, annotation_text="no difference",
                  annotation_font_size=11, annotation_font_color=INK2)
    fig.update_xaxes(title_text="difference in TA_within (family-block bootstrap)", row=1, col=2)
    _legend_below(_style(fig, 620))
    sup = decision["H1"]
    return rec("h1", "Primary test of tree agreement within groups", "positive" if sup else "negative",
               f"Family-block criterion {'met' if sup else 'unmet'}, max-T p {_p(g['models'][run]['p'])}",
               f"Distinct final name words, layer {decision['layer']}. Left: node-averaged triplet accuracy. "
               "Right: paired differences against each required control. Bars show percentile intervals from "
               "resampling families at fixed modal layers. The paper also reports order-block sensitivity.", fig, (1150, 640))


def fig_depth_profile(run) -> dict | None:
    s = _load(RESULTS_DIR / run / "stage1b" / "results.json")
    if not s:
        return None
    dp, cnt = s["depth_profile"], s["depth_counts"]
    lv = [k for k in RANK_LABELS if cnt.get(k, 0) >= 30]  # sparse levels (e.g. 3 same-genus triplets) are noise
    x = [RANK_LABELS[k] for k in lv]
    fig = go.Figure()
    for r, dash in [("qwen35_2b", "solid"), ("qwen35_2b_random", "solid"), ("pythia410m_step0", "dot"),
                    ("char3", "solid"), ("rank_oracle", "dash"), ("class_oracle", "dot")]:
        if r not in dp:
            continue
        cat, color, sym = CATEGORY[r]
        y = [dp[r].get(k) for k in lv]
        fig.add_scatter(x=x, y=y, mode="lines+markers", name=DISPLAY[r], line={"color": color, "dash": dash,
                                                                              "width": 3 if r == run else 2},
                        marker={"symbol": sym, "size": 9, "color": color},
                        customdata=[cnt[k] for k in lv],
                        hovertemplate=f"<b>{DISPLAY[r]}</b><br>%{{x}}<br>accuracy %{{y:.3f}}<br>%{{customdata}} "
                                      "triplets<extra></extra>")
    fig.add_hline(1.0, line_color=AQUA, line_width=1, annotation_text="exact tree (ground truth) = 1",
                  annotation_font_color=AQUA, annotation_font_size=11)
    _chance(fig)
    fig.update_yaxes(title_text="triplet accuracy", range=[0, 1.05])
    fig.update_xaxes(title_text="deepest rank shared by all three species (left: deep splits, right: recent splits)")
    _style(fig, 540)
    return rec("depth", "Where on the tree is the model right? Accuracy by depth of the split", "mixed",
               "largest gain between classes, smaller inside orders",
               "All resolved triplets grouped by the deepest Linnaean rank the three species share (n in the hover; "
               "levels with fewer than 30 triplets are omitted). The model is best for triplets inside one phylum split "
               "between classes, where the class-only oracle also scores well; its margin over the untrained networks "
               "shrinks inside classes and orders, and every level stays far below the exact tree.", fig)


def fig_per_stratum(run) -> dict | None:
    from phylogeo.viz.pca import STRATUM_COLORS

    s = _load(RESULTS_DIR / run / "stage1b" / "results.json")
    if not s:
        return None
    ps = s["exploratory"]["per_stratum_ci95"]
    items = sorted(ps.items(), key=lambda kv: kv[1]["ta"])
    fig = go.Figure()
    for st, v in items:
        lo, hi = v["ci95"]
        fig.add_scatter(x=[v["ta"]], y=[f"{st} ({v['n_nodes']} node{'s' if v['n_nodes'] != 1 else ''})"], mode="markers", showlegend=False,
                        marker={"color": STRATUM_COLORS.get(st, GRAY), "size": 6 + min(v["n_nodes"], 40) / 3},
                        error_x={"type": "data", "symmetric": False, "array": [hi - v["ta"]],
                                 "arrayminus": [v["ta"] - lo], "color": STRATUM_COLORS.get(st, GRAY), "thickness": 2},
                        hovertemplate=f"<b>{st}</b><br>TA_within %{{x:.3f}}<br>95% CI {lo:.3f}–{hi:.3f}<br>"
                                      f"{v['n_nodes']} decision nodes<extra></extra>")
    _chance(fig, "x")
    fig.update_xaxes(title_text="TA_within (family-block bootstrap 95% CI; marker size = number of decision nodes)",
                     range=[-0.02, 1.02])
    _style(fig, 560)
    clear = [st for st, v in ps.items() if v["ci95"][0] > 1 / 3]
    below = [st for st, v in ps.items() if v["ci95"][1] < 1 / 3]
    verdict = "clear in " + (", ".join(sorted(clear)) or "no stratum") + (
        "; below chance in " + ", ".join(f"{b} ({ps[b]['n_nodes']} nodes)" for b in sorted(below)) if below else "")
    return rec("strata", "Which groups carry the signal?", "mixed", verdict,
               "TA_within per stratum with family-block bootstrap CIs. Colour = super-group (blue vertebrates, orange "
               "arthropods, aqua other invertebrates). Only strata whose whole CI lies above chance show clear "
               "within-group tree order; small strata have very wide CIs.", fig, (1100, 580))


def fig_layer_profile(run) -> dict | None:
    s = _load(RESULTS_DIR / run / "stage1b" / "results.json")
    if not s:
        return None
    fig = go.Figure()
    main = {"qwen35_2b": (BLUE, "solid", 3), "qwen35_2b_random": (GRAY, "solid", 2),
            "pythia410m_step0": (GRAY, "dot", 2), "pythia410m": (BLUE, "dot", 2)}
    replay = _load(RESULTS_DIR / "scientific-verification/geometry.json")["models"]
    scale = _load(RESULTS_DIR / "scientific-verification/scale.json")
    for r, rec_ in s["models"].items():
        y = replay[r]["by_layer"] if r in replay else scale[r]["by_layer"]
        color, dash, w = main.get(r, (GRAY, "solid", 1))
        fig.add_scatter(x=[i / (len(y) - 1) for i in range(len(y))], y=y, mode="lines+markers", name=DISPLAY.get(r, r),
                        line={"color": color, "dash": dash, "width": w}, marker={"size": 5, "color": color},
                        customdata=list(range(len(y))),
                        hovertemplate=f"<b>{DISPLAY.get(r, r)}</b><br>layer %{{customdata}}<br>TA_within %{{y:.3f}}"
                                      "<extra></extra>")
    for f in sorted(RESULTS_DIR.glob("*/stage5/scale.json")):
        r5 = json.loads(f.read_text())
        if r5["run"] in s["models"]:
            continue
        y = scale[r5["run"]]["by_layer"]
        fig.add_scatter(x=[i / (len(y) - 1) for i in range(len(y))], y=y, mode="lines", name=r5["run"],
                        legendgroup="stage 5", legendgrouptitle_text="stage-5 models (click to show)",
                        line={"color": LIGHTGRAY, "width": 1.5}, visible="legendonly",
                        hovertemplate=f"<b>{r5['run']}</b><br>TA_within %{{y:.3f}}<extra></extra>")
    _chance(fig)
    fig.update_xaxes(title_text="relative layer depth (0 = embeddings, 1 = last layer)")
    fig.update_yaxes(title_text="TA_within")
    _style(fig, 520).update_layout(legend={"font": {"size": 11}})
    y = replay[run]["by_layer"]
    peak = int(np.argmax(y))
    return rec("layers", "Across layers: when does tree order appear?", "positive",
               f"peaks at layer {peak} of {len(y) - 1} ({peak / (len(y) - 1):.0%} depth)",
               "TA_within at every layer for the model and its controls (stage 1b, 1000-permutation max-T test). "
               "Stage-5 models are hidden at first; click their legend entries to compare.", fig, (1100, 540))


# ======================================================================================
# Negative and mixed results
# ======================================================================================

def fig_secondary(run) -> dict | None:
    s = _load(RESULTS_DIR / run / "stage1b" / "results.json")
    if not s:
        return None
    g = _load(RESULTS_DIR / "scientific-verification/geometry.json")
    rows = [("S2", "S2: family/genus split within order"), ("S3", "S3: uniquely wrong spelling choice"),
            ("S4", "S4: uniquely wrong GloVe choice"), ("no_shared_trigram", "no shared trigram (exploratory)")]
    fig = go.Figure()
    for i, (k, label) in enumerate(rows):
        row = g[k] if k == "no_shared_trigram" else g["secondary"][k]
        group = row["comparison"] if k == "no_shared_trigram" else row["comparisons"]
        v = {name: result["ta"] for name, result in group.items()}
        lo, hi = group[run]["ci"]
        ylab = f"{label}<br>{row['triplets']} triplets"
        fig.add_scatter(x=[v["qwen35_2b_random"], v["pythia410m_step0"], v["qwen35_2b"]], y=[ylab] * 3, mode="lines",
                        line={"color": LIGHTGRAY, "width": 6}, showlegend=False, hoverinfo="skip")
        for r in ["qwen35_2b_random", "pythia410m_step0", "qwen35_2b"]:
            cat, color, sym = CATEGORY[r]
            err = ({"type": "data", "symmetric": False, "array": [hi - v[r]], "arrayminus": [v[r] - lo], "color": color,
                    "thickness": 2} if r == "qwen35_2b" else None)
            fig.add_scatter(x=[v[r]], y=[ylab], mode="markers", name=DISPLAY[r], legendgroup=r, showlegend=(i == 0),
                            marker={"color": color, "symbol": sym, "size": 13 if r == "qwen35_2b" else 11,
                                    "line": {"color": SURFACE, "width": 2}}, error_x=err,
                            hovertemplate=f"<b>{DISPLAY[r]}</b><br>{label}<br>accuracy %{{x:.3f}}<extra></extra>")
    _chance(fig, "x")
    fig.update_xaxes(title_text="triplet accuracy (model with family-block 95% CI)")
    fig.update_yaxes(autorange="reversed")
    _legend_below(_style(fig, 520)).update_layout(margin={"l": 290})
    return rec("secondary", "Within-order and lexical subsets", "mixed", "Conditional comparisons",
               "Points compare the trained model with both untrained networks at their modal layers. "
               "The model interval resamples families. The table gives multiplicity-adjusted tests. "
               "The final subset conditions on names sharing no trigram.", fig, (1100, 520))


def fig_conflict(run) -> dict | None:
    if run != "qwen35_2b":
        return None
    c = _load(RESULTS_DIR / "scientific-verification/outcomes.json")["conflict"]
    reps = list(c["summary"])
    trip = [t for t in c["rows"] if t["resolved"]]
    ylab = [f"{t['anchor']}: <b>{t['closer']}</b> or {t['other']}?" for t in trip]
    z = [[t["choices"][k] for k in reps] for t in trip]
    labels = [[{1: "tree", 0: "tie", -1: "other", None: "missing"}[v] for v in row] for row in z]
    xlab = [f"{DISPLAY[k]}<br>{c['summary'][k]['win']} tree · {c['summary'][k]['tie']} tied" for k in reps]
    fig = go.Figure(go.Heatmap(z=z, x=xlab, y=ylab, zmin=-1, zmax=1, xgap=2, ygap=2,
                              colorscale=[[0, ORANGE], [.25, ORANGE], [.25, GRAY], [.75, GRAY], [.75, BLUE], [1, BLUE]],
                              showscale=False, text=labels, texttemplate="%{text}", textfont={"size": 10, "color": "white"},
                              hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont={"size": 11})
    fig.update_xaxes(side="top", tickfont={"size": 11})
    _style(fig, 60 + 24 * len(trip), legend=False).update_layout(margin={"l": 330, "t": 90, "b": 20})
    s = c["summary"][run]
    return rec("conflict", "Tree choices on hand-picked conflict items", "example",
               f"{s['win']} tree, {s['tie']} tied, {s['loss']} other",
               "Bold names are the closer partners on the frozen topology. Colors distinguish the two strict choices "
               "and equal distances. Species recur across items and overlap the primary panel.", fig, (1100, 80 + 24 * len(trip)))


def fig_probe(run) -> dict | None:
    s2 = _load(RESULTS_DIR / run / "stage2" / "results.json")
    if not s2:
        return None
    pr = s2["probe"]
    reps = [("model", "Qwen3.5-2B", BLUE), ("shuffled_tree", "shuffled tree", GRAY), ("random_init", "random weights", GRAY),
            ("glove", "GloVe input", ORANGE), ("class_blob", "class labels only", AQUA)]
    fig = make_subplots(1, 2, subplot_titles=["A. All held-out pairs: Pearson r", "B. Inside strata: mean Spearman"],
                        horizontal_spacing=0.12)
    for col, key in [(1, "test_test_pearson"), (2, "test_test_within_stratum_spearman")]:
        rs = [r for r in reps if key in pr.get(r[0], {})]
        fig.add_bar(x=[r[1] for r in rs], y=[pr[r[0]][key][0] for r in rs],
                    error_y={"type": "data", "array": [pr[r[0]][key][1] for r in rs], "color": INK2},
                    marker={"color": [r[2] for r in rs], "line": {"color": SURFACE, "width": 2}}, showlegend=False,
                    hovertemplate="<b>%{x}</b><br>%{y:.3f} (± SD over seeds)<extra></extra>", row=1, col=col)
    fig.add_hline(pr["class_only_predictor_pearson_all_pairs"], line_dash="dash", line_color=AQUA, row=1, col=1,
                  annotation_text="class-only predictor", annotation_font_color=AQUA, annotation_font_size=11)
    raw = _load(RESULTS_DIR / "scientific-verification/geometry.json")["raw_probe_comparison"]["mean"]
    fig.add_hline(raw, line_dash="dash", line_color=BLUE, row=1, col=2,
                  annotation_text="raw distances (same held-out pairs)",
                  annotation_font_color=BLUE, annotation_font_size=11)
    fig.update_yaxes(title_text="held-out families (5-fold)", row=1, col=1)
    _style(fig, 520, legend=False)
    beaten = pr["class_blob"]["test_test_pearson"][0] >= pr["model"]["test_test_pearson"][0]
    return rec("probe", "Goodfire-style 10-d flat probe: tree or coarse groups?", "negative" if beaten else "positive",
               "class-only input has higher mean r" if beaten else "model has higher mean r",
               "Probe trained to predict tree distance from representations, evaluated on held-out families (error bars "
               "= SD over seeds). The class-centroid-plus-noise control produces a high global correlation. "
               "These historical fits used a full-panel layer choice and target scale. Inside strata the probe "
               + ("has a lower mean rank correlation than the raw distances on the same pairs."
                  if raw > pr["model"]["test_test_within_stratum_spearman"][0] else "adds little over raw distances."),
               fig, (1100, 540))


def fig_treeness(run) -> dict | None:
    s2 = _load(RESULTS_DIR / run / "stage2" / "results.json")
    if not s2:
        return None
    d, nj = s2["delta_rel"], s2["nj_rf"]
    fig = make_subplots(1, 2, horizontal_spacing=0.14,
                        subplot_titles=["A. Sampled base-point relative hyperbolicity",
                                        "B. Neighbour-joining tree vs reference (normalized RF)"])
    a = [("exact tree", d["tree_oracle"], 0, AQUA), ("class clusters", d["class_blob_mean"], d["class_blob_sd"], ORANGE),
         ("Qwen3.5-2B", d["model"], 0, BLUE), ("matched Gaussian", d["matched_gaussian_mean"], d["matched_gaussian_sd"], GRAY)]
    b = [("exact tree", 0.0, 0, AQUA), ("Qwen3.5-2B", nj["model"], 0, BLUE),
         ("class-blob inputs", nj["class_blob_null_mean"], nj["class_blob_null_sd"], ORANGE),
         ("shuffled labels", nj["perm_null_mean"], nj["perm_null_sd"], GRAY)]
    for col, items in [(1, a), (2, b)]:
        fig.add_bar(x=[i[0] for i in items], y=[i[1] for i in items],
                    error_y={"type": "data", "array": [i[2] for i in items], "color": INK2},
                    marker={"color": [i[3] for i in items], "line": {"color": SURFACE, "width": 2}},
                    text=[f"{i[1]:.3f}" for i in items], textposition="outside", showlegend=False,
                    hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>", row=1, col=col)
    fig.update_yaxes(title_text="lower = more tree-like", row=1, col=1)
    _style(fig, 500, legend=False)
    return rec("treeness", "Is the geometry tree-like (hyperbolic)?", "negative", "between clusters and noise",
               "Both measures are 0 for an exact tree (aqua, ground truth). The model's δ_rel lies between pure class "
               "clusters and a spectrum-matched Gaussian, so it does not separate trees from clusters; its "
               "neighbour-joining tree is closer to the reference than random labels or class-blob inputs, but far "
               "from exact (error bars = SD over null draws).", fig, (1100, 520))


def fig_causal(run) -> dict | None:
    s3 = _load(RESULTS_DIR / run / "stage3" / "results.json")
    if not s3 or "class_control" not in s3:
        return None
    a, w, cc = s3["C_abl"], s3["C_swap"], s3["class_control"]
    audit = _load(RESULTS_DIR / "scientific-verification/outcomes.json")["causal"]["expanded"]
    fig = make_subplots(1, 2, horizontal_spacing=0.12,
                        subplot_titles=[f"A. Ablation: rank p {_p(audit['ablation_rank'])}",
                                        f"B. Post-edit donor prevalence: rank p {_p(audit['donor_prevalence_rank'])}"])
    fig.add_histogram(x=a["random_mean_drops"], nbinsx=25, marker={"color": GRAY, "line": {"color": SURFACE, "width": 1}},
                      name=f"{a['n_random_subspaces']} random subspaces", legendgroup="rand",
                      hovertemplate="drop %{x}<br>%{y} subspaces<extra>random</extra>", row=1, col=1)
    fig.add_histogram(x=w["switch_rate_random"], marker={"color": GRAY, "line": {"color": SURFACE, "width": 1}},
                      showlegend=False, legendgroup="rand", xbins={"size": 1 / w["n_pairs"]},
                      hovertemplate="donor prevalence %{x}<br>%{y} subspaces<extra>random</extra>", row=1, col=2)
    ymax = {1: np.histogram(a["random_mean_drops"], bins=25)[0].max(),
            2: np.unique(np.round(np.array(w["switch_rate_random"]) * w["n_pairs"]), return_counts=True)[1].max()}
    for col, val, name, color, dash in [(1, a["mean_drop_logp_probe"], "phylogeny probe subspace", BLUE, "solid"),
                                        (1, cc["mean_drop_logp_class"], "class-only subspace", ORANGE, "dash"),
                                        (2, w["switch_rate_probe"], "phylogeny probe subspace", BLUE, "solid"),
                                        (2, cc["switch_rate_class"], "class-only subspace", ORANGE, "dash"),
                                        (2, w["switch_rate_no_edit"], "no edit", MUTED, "dot")]:
        fig.add_scatter(x=[val, val], y=[0, 1.08 * ymax[col]], mode="lines", name=name, legendgroup=name,
                        showlegend=(col == 1 or name == "no edit"), line={"color": color, "width": 3, "dash": dash},
                        hovertemplate=f"<b>{name}</b><br>%{{x:.3f}}<extra></extra>", row=1, col=col)
    for col, vals in [(1, list(a["random_mean_drops"]) + [a["mean_drop_logp_probe"], cc["mean_drop_logp_class"]]),
                      (2, list(w["switch_rate_random"]) + [w["switch_rate_probe"], cc["switch_rate_class"]])]:
        pad = 0.08 * (max(vals) - min(vals))
        fig.update_xaxes(range=[min(vals) - pad, max(vals) + pad], row=1, col=col)
    fig.update_xaxes(title_text="mean drop in log P(correct class) [nats]", row=1, col=1)
    fig.update_xaxes(title_text="share of pairs predicting the donor class", row=1, col=2)
    fig.update_yaxes(title_text="random subspaces", row=1, col=1)
    _legend_below(_style(fig, 520)).update_layout(barmode="overlay")
    low, high = audit["new_donor_bounds"]
    return rec("causal", "Ablation and donor-class predictions", "mixed", "Registered switch comparison unevaluable",
               f"Gray bars are random-subspace results. The probe and class-only edits are shown against those draws. "
               f"Donor answers changed from {audit['before']} before to {audit['after']} after the probe edit. "
               f"Between {low} and {high} were newly donor answers. The historical random-swap distribution "
               "measures post-edit prevalence.", fig, (1100, 520))


def fig_habitat(run) -> dict | None:
    e = _load(RESULTS_DIR / run / "eco" / "results.json")
    if not e:
        return None
    hs = e["heldout_scores"]
    mam = [h for h in hs if h["stratum"] == "mammals" and h.get("forstrat_mammal")]
    bird = [h for h in hs if h["stratum"] == "birds" and h.get("water_pct_bird") not in ("", None)]
    t = e["tests"]
    family = _load(RESULTS_DIR / "scientific-verification/outcomes.json")["habitat"]["holm"]
    lab = {"M": "marine", "G": "ground", "Ar": "arboreal", "A": "aerial", "S": "scansorial"}
    fig = make_subplots(1, 2, horizontal_spacing=0.12,
                        subplot_titles=[f"A. Held-out mammals by foraging stratum (marine < ground: Holm p "
                                        f"{_p(family['M1'])})",
                                        f"B. Held-out birds: water foraging (ρ = {t['B1_water_negative']['rho']:.2f}, "
                                        f"Holm p {_p(family['B1'])})"])
    cats = sorted({h["forstrat_mammal"] for h in mam}, key=lambda k: np.median([h["axis_score"] for h in mam
                                                                             if h["forstrat_mammal"] == k]))
    for k in cats:
        g = [h for h in mam if h["forstrat_mammal"] == k]
        fig.add_box(y=[h["axis_score"] for h in g], name=f"{lab.get(k, k)} (n={len(g)})", boxpoints="all", jitter=0.4,
                    pointpos=0, marker={"color": ORANGE if k == "M" else BLUE, "size": 7},
                    line={"color": ORANGE if k == "M" else BLUE}, fillcolor="rgba(0,0,0,0)",
                    text=[h["common_name"] for h in g], hovertemplate="<b>%{text}</b><br>axis score %{y:.2f}<extra></extra>",
                    showlegend=False, row=1, col=1)
    fig.add_scatter(x=[float(h["water_pct_bird"]) for h in bird], y=[h["axis_score"] for h in bird], mode="markers",
                    marker={"color": BLUE, "size": 8, "line": {"color": SURFACE, "width": 1}},
                    text=[h["common_name"] for h in bird], showlegend=False,
                    hovertemplate="<b>%{text}</b><br>water foraging %{x}%<br>axis score %{y:.2f}<extra></extra>",
                    row=1, col=2)
    fig.update_yaxes(title_text=f"layer-{e['layer']} PC1 score (low = aquatic end)", row=1, col=1)
    fig.update_xaxes(title_text="foraging in or around water (EltonTraits, %)", row=1, col=2)
    _style(fig, 520, legend=False)
    return rec("habitat", "Foraging traits on held-out species", "mixed", "Aquatic associations in mammals and birds",
               "The axis and its direction were chosen on discovery species. The labels show four-slot Holm p-values "
               "for species-level tests. The table also gives within-order sensitivity. The aerial bird criterion was "
               "unmet and the mammal sample contained no aerial group.", fig, (1100, 540))


def fig_scale(run) -> dict | None:
    s5 = _load(RESULTS_DIR / "stage5" / "summary.json")
    if not s5:
        return None
    scale = _load(RESULTS_DIR / "scientific-verification/scale.json")
    runs = [dict(r, crossfit_ta_within=scale[r["run"]]["crossfit"], maxT_p=scale[r["run"]]["p"]) for r in s5["runs"]]
    geometry = _load(RESULTS_DIR / "scientific-verification/geometry.json")
    fig = make_subplots(1, 2, horizontal_spacing=0.1, subplot_titles=["A. Model size (cross-fitted TA_within)",
                                                                       "B. Training dynamics: Pythia-1.4B checkpoints"])
    fams = [("Qwen3.5", lambda m: "qwen3.5" in m, BLUE, "circle", "lines+markers+text"),
            ("Qwen3 (full attention)", lambda m: "qwen3-" in m, BLUE, "diamond-open", "markers+text"),
            ("Pythia (the Pile)", lambda m: "pythia" in m, ORANGE, "square", "lines+markers+text")]
    for fam, match, color, sym, mode in fams:
        rs = sorted([r for r in runs if r["step"] is None and match(r["model_id"].lower())], key=lambda r: r["params"])
        fig.add_scatter(x=[r["params"] for r in rs], y=[r["crossfit_ta_within"] for r in rs], mode=mode,
                        name=fam, line={"color": color, "width": 2},
                        marker={"color": color, "symbol": sym, "size": 10}, text=[r["run"] for r in rs],
                        textposition="top center", textfont={"size": 10, "color": INK2},
                        hovertemplate="<b>%{text}</b><br>%{x:.2s} parameters<br>TA_within %{y:.3f}<extra></extra>",
                        row=1, col=1)
    ck = sorted([r for r in runs if r["step"] is not None], key=lambda r: r["step"])
    fig.add_scatter(x=[max(r["step"], 0.5) for r in ck], y=[r["crossfit_ta_within"] for r in ck], mode="lines+markers",
                    name="Pythia-1.4B checkpoints", line={"color": ORANGE, "width": 2},
                    marker={"color": ORANGE, "size": 9, "symbol": "square-open"},
                    customdata=[[r["step"], r["maxT_p"]] for r in ck],
                    hovertemplate="step %{customdata[0]}<br>TA_within %{y:.3f}<br>max-T p %{customdata[1]}<extra></extra>",
                    row=1, col=2)
    for name in s5["reference_ta_within"]:
        if name not in geometry["family"]:
            continue
        v = geometry["models"][name]["crossfit"] if name in geometry["models"] else geometry["family"][name]["ta"]
        fig.add_hline(v, line_dash="dot", line_color=LIGHTGRAY, row=1, col=1, annotation_text=DISPLAY.get(name, name),
                      annotation_font_size=10, annotation_font_color=MUTED, annotation_position="bottom right")
    for col in (1, 2):
        _chance(fig, "y", 1, col)
    fig.update_xaxes(type="log", title_text="parameters (nominal, log)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="training step (log; step 0 plotted at 0.5)", row=1, col=2)
    fig.update_yaxes(title_text="cross-fitted TA_within", row=1, col=1)
    _legend_below(_style(fig, 540))
    return rec("scale", "Model size and checkpoint trajectories", "example", "Descriptive comparisons",
               "Each point uses saved activations, the written tie rule and family cross-fitting. Step zero is offset "
               "to show it on the log scale. Checkpoints share one training trajectory. Size, data, architecture and "
               "training history can covary across final models.", fig, (1150, 560))


def fig_sensitivity(run) -> dict | None:
    s6 = _load(RESULTS_DIR / run / "stage6" / "results.json")
    if not s6:
        return None
    refs = [r for r in s6["representations"] if r != run]
    n = s6["n_rep"]

    def best(blk):
        return [max(blk[k]["values"][i] for k in refs) for i in range(n)]

    rows = [("triplet sampling seed", s6["triplet_seed"][run]["values"], best(s6["triplet_seed"])),
            ("80% species subsample", s6["species_subsample"][run]["values"], best(s6["species_subsample"])),
            ("template halves (4 of 8)", s6["templates"]["half"]["ta_within"]["values"], None),
            ("single template", s6["templates"]["single"]["ta_within"]["values"], None),
            ("cross-fit family splits", s6["crossfit_seed"]["crossfit_ta_within"]["values"], None)]
    fig = go.Figure()
    rng = np.random.default_rng(0)  # jitter only (display)
    for i, (lab, vals, ctrl) in enumerate(rows):
        for v, name, color, sym, show in [(vals, "Qwen3.5-2B", BLUE, "circle", i == 0),
                                          (ctrl, "best control on the same draw", GRAY, "square", i == 0)]:
            if v is None:
                continue
            fig.add_scatter(x=v, y=i + rng.uniform(-0.15, 0.15, len(v)), mode="markers", name=name, legendgroup=name,
                            showlegend=show, marker={"color": color, "symbol": sym, "size": 8,
                                                     "line": {"color": SURFACE, "width": 1}},
                            hovertemplate=f"<b>{lab}</b><br>{name}<br>TA_within %{{x:.3f}}<extra></extra>")
    _chance(fig, "x")
    fig.add_vline(s6["main_run"][run], line_color=BLUE, line_width=1, line_dash="dot", annotation_text="main run",
                  annotation_font_color=BLUE, annotation_font_size=11)
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=[r[0] for r in rows], autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text=f"TA_within at layer {s6['layer']} (one dot per repetition)")
    _legend_below(_style(fig, 480))
    return rec("sensitivity", "Sensitivity: seeds, templates, species subsamples", "example", "Archived perturbations",
               f"{n} repetitions per perturbation. On every draw where controls were recomputed, the model stays above "
               "the largest of the controls included on that particular draw. The control set varies by perturbation.", fig, (1100, 500))


# ======================================================================================
# Output examples
# ======================================================================================

def _neighbors(run):
    d = _load(RESULTS_DIR / run / "viz" / "neighbors.json")
    ex = _load(RESULTS_DIR / run / "examples" / "neighbors.json")
    return d, ex


def fig_neighbor_levels(run) -> dict | None:
    d, _ = _neighbors(run)
    if not d:
        return None
    reps = ["tree", run, *[r for r in d["levels"] if r not in ("tree", run)]]
    fig = go.Figure()
    for lv, name in enumerate(d["level_names"]):
        shares = [float((np.array(d["levels"][r]) == lv).mean()) for r in reps]
        fig.add_bar(y=[DISPLAY.get(r, r) for r in reps], x=shares, orientation="h", name=name,
                    marker={"color": LEVEL_COLORS[lv], "line": {"color": SURFACE, "width": 2}},
                    text=[f"{100 * s:.0f}%" if s >= 0.05 else "" for s in shares], textposition="inside",
                    insidetextfont={"color": "white" if lv < 3 else INK, "size": 11},
                    hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.1%}}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", title_text=f"share of each species' {d['k']} nearest neighbours")
    fig.update_yaxes(autorange="reversed")
    _legend_below(_style(fig, 400), traceorder="normal").update_layout(
                                   margin={"l": 230})
    return rec("neighbor_levels", "Nearest neighbours: how closely related are they?", "positive",
               "far more same-order neighbours than untrained",
               f"For every species, its {d['k']} nearest species at layer {d['layer']}, coloured by the closest shared "
               "rank (dark = same genus ... light = different phylum). The tree row is the ground-truth ceiling (ties in "
               "tree distance broken by table order). Descriptive: names that share a head noun (e.g. two sharks) are not "
               "excluded here, unlike in the confirmatory metric.", fig, (1100, 400))


def fig_neighbor_examples(run, n_show: int = 5) -> dict | None:
    d, ex = _neighbors(run)
    if not d or not ex:
        return None
    idx = {n: i for i, n in enumerate(d["names"])}
    queries = [q["name"] for q in ex["queries"] if q["name"] in idx]
    reps = [("tree", "Ground truth: closest on the tree"), (run, "Qwen3.5-2B"),
            ([r for r in d["levels"] if r not in ("tree", run)][0], "Untrained Qwen3.5-2B")]
    fig = make_subplots(1, 3, subplot_titles=[t for _, t in reps], horizontal_spacing=0.02, shared_yaxes=True)
    scale = [[i / (len(LEVEL_COLORS) - 1), c] for i, c in enumerate(LEVEL_COLORS)]
    for col, (r, _) in enumerate(reps, start=1):
        Z = [[d["levels"][r][idx[q]][j] for j in range(n_show)] for q in queries]
        T = [[d["names"][d["neighbors"][r][idx[q]][j]] for j in range(n_show)] for q in queries]
        Lt = [[d["level_names"][z] for z in row] for row in Z]
        fig.add_heatmap(z=Z, x=[f"#{j + 1}" for j in range(n_show)], y=queries, colorscale=scale, zmin=0,
                        zmax=len(LEVEL_COLORS) - 1, showscale=False, xgap=2, ygap=2, customdata=Lt, text=T,
                        hovertemplate="query <b>%{y}</b><br>neighbour %{x}: <b>%{text}</b><br>%{customdata}<extra></extra>",
                        row=1, col=col)
        for qi, row in enumerate(Z):  # per-cell labels: white on dark steps, ink on light ones
            for j, z in enumerate(row):
                lab = T[qi][j] if len(T[qi][j]) <= 15 else T[qi][j][:14] + "…"
                fig.add_annotation(x=j, y=qi, xref=f"x{col if col > 1 else ''}", yref="y", text=lab,
                                   showarrow=False, font={"size": 9, "color": "white" if z < 3 else INK})
    for lv, name in enumerate(d["level_names"]):  # legend entries for the ordinal colour key
        fig.add_scatter(x=[None], y=[None], mode="markers", name=name, marker={"color": LEVEL_COLORS[lv], "size": 12,
                                                                                "symbol": "square"})
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=False, title_text="neighbour rank")
    _legend_below(_style(fig, 160 + 34 * len(queries)), bottom=110).update_layout(
                                                         margin={"l": 150, "t": 60})
    return rec("neighbor_examples", "Output examples: the 5 nearest species", "example",
               "rule-chosen queries (one per stratum)",
               "Query species chosen by a fixed rule (the stage-report examples). Each cell is a neighbour, coloured by "
               "its closest shared rank with the query; compare the model with the tree (ground truth) and with the "
               "untrained network.", fig, (1400, 170 + 34 * len(queries)))


# ======================================================================================

# ======================================================================================
# Animated 3-D top-3 PCA (stage 1a look, every layer)
# ======================================================================================

def _clip(P: np.ndarray, lim: float) -> np.ndarray:
    """Positions outside the shared box -> NaN (not drawn; Plotly serializes NaN as a gap)."""
    P = np.array(P, dtype=float)
    P[(np.abs(P) > lim).any(1)] = np.nan
    return P


def n_outside(P, lim: float) -> int:
    """Number of species outside the shared box (reported in the panel title, never silently dropped)."""
    return int((np.abs(np.asarray(P)) > lim).any(1).sum())


TREE_GRAY = "#b9b8b2"  # recessive line colour for tree edges that span several highlighted clades


def _tree_lines(tree, pos: np.ndarray, scene: str, nodes, color: str = TREE_GRAY, name: str = "reference tree",
                group: str = "tree", showlegend: bool = False, width: float = 1.5, opacity: float = 0.5):
    """One Scatter3d line trace: the edge from each node in ``nodes`` to its parent, at positions ``pos`` [N, 3]
    (None breaks the line between edges, so one trace draws many segments)."""
    xs, ys, zs = _segments(tree, pos, nodes)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line={"color": color, "width": width}, opacity=opacity,
                        name=name, legendgroup=group, showlegend=showlegend, hoverinfo="skip", scene=scene)


def _segments(tree, pos: np.ndarray, nodes) -> list[list]:
    """Per coordinate, the polyline (parent, child, None, ...) that draws the edge above each node in ``nodes``."""
    return [[c for v in nodes for c in (pos[tree.parent[v], k], pos[v, k], None)] for k in range(pos.shape[1])]


def _point_traces(coords: np.ndarray, names, labels, style: dict, scene: str, showlegend: bool, size: float = 3.5,
                  detail=None) -> list:
    """One Scatter3d marker trace per label in ``style`` (label -> (colour, symbol)), in style order; labels with
    no points are skipped. Hover shows the name and ``detail`` (per point) or the label. Legend groups are keyed
    by label, so one legend click toggles a label in every panel."""
    labels, names = np.asarray(labels), np.asarray(names)
    detail = labels if detail is None else np.asarray(detail)
    out = []
    for s, (color, sym) in style.items():
        m = labels == s
        if m.any():
            out.append(go.Scatter3d(x=coords[m, 0], y=coords[m, 1], z=coords[m, 2], mode="markers", name=s,
                                    legendgroup=s, showlegend=showlegend, text=names[m], customdata=detail[m],
                                    scene=scene, marker={"color": color, "symbol": sym, "size": size,
                                                         "line": {"color": SURFACE, "width": 0.5}},
                                    hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>"))
    return out


def _cloud_traces(coords: np.ndarray, names, strata, tree, scene: str, showlegend: bool, lim: float) -> list:
    """Scatter3d traces for one panel: the reference tree drawn with internal nodes at the centroid of their
    descendant leaves (as in stage 1a, computed from all species), then one marker trace per stratum
    (colour = super-group). Points outside the shared box are not drawn. Legend groups are shared across
    panels, so one legend click toggles a stratum in both."""
    from phylogeo.viz.pca import STRATUM_STYLE, overlay_positions

    pos = _clip(overlay_positions(tree, coords), lim)
    seg = [v for v in range(tree.N) if v != tree.root]
    return [_tree_lines(tree, pos, scene, seg, name="reference tree (centroids)", showlegend=showlegend),
            *_point_traces(_clip(coords, lim), names, strata, {s: (c, sym) for s, (c, _, sym) in STRATUM_STYLE.items()},
                           scene, showlegend)]


def _pair_fig(panels: list[tuple[np.ndarray, str]], names, strata, tree, lim: float, eye=(1.3, 0.8, 0.6)) -> go.Figure:
    """Static side-by-side 3-D snapshot (PNG / GIF frames); ``panels`` = [(coords, title), ...], one shared box."""
    fig, k = go.Figure(), len(panels)
    ax = {"range": [-lim, lim], "showticklabels": False, "backgroundcolor": SURFACE, "gridcolor": GRID,
          "zerolinecolor": AXIS}
    for i, (C, title) in enumerate(panels):
        sc = "scene" if i == 0 else f"scene{i + 1}"
        fig.add_traces(_cloud_traces(C, names, strata, tree, sc, showlegend=(i == 0), lim=lim))
        fig.update_layout({sc: {"domain": {"x": [i / k + 0.005, (i + 1) / k - 0.005], "y": [0.04, 0.92]},
                                "xaxis": {**ax, "title": {"text": "axis 1"}}, "yaxis": {**ax, "title": {"text": "axis 2"}},
                                "zaxis": {**ax, "title": {"text": "axis 3"}}, "aspectmode": "cube",
                                "camera": {"eye": dict(zip("xyz", eye))}}})
        fig.add_annotation(text=title, x=(i + 0.5) / k, y=0.995, xref="paper", yref="paper", showarrow=False,
                           xanchor="center", yanchor="top", font={"size": 13, "color": INK})
    _style(fig, 640).update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0})
    return _legend_below(fig, bottom=70, itemsizing="constant", font={"size": 11})


def _lims(coords) -> list[float]:
    """Shared half-width of the 3-D box at each layer: the trained model's largest |coordinate| at that layer,
    plus 3%. Every trained-model species stays inside; in the untrained network the few outlier names that fall
    outside are counted in the panel title instead of shrinking everything else to a dot."""
    return [round(1.03 * float(np.abs(np.asarray(C)).max()), 2) for C in coords]


def _lim_at(lims: list[float], t: float) -> float:
    """Box half-width between layers (linear, like the positions)."""
    if len(lims) == 1:
        return lims[0]
    k = min(int(t), len(lims) - 2)
    return lims[k] + (lims[k + 1] - lims[k]) * (t - k)


def _player_html(d: dict, run: str, control: str) -> str:
    """Self-contained animated side-by-side 3-D player (vanilla JS + Plotly.js, loaded once in the page head).
    Left: the trained model. Right: its random-init twin (or the expected tree shape). One layer clock drives both
    panels, positions are interpolated linearly between consecutive layers, the tree overlay (internal node = mean
    of descendant leaves) is recomputed every frame, and the two cameras are linked (auto-rotation or dragging
    either panel turns both)."""
    from phylogeo.viz.pca import STRATUM_STYLE

    reps = {k: {"coords": np.round(np.array(v["coords"]), 2).tolist(), "evr": v["evr"]} for k, v in d["reps"].items()}
    style = {s: [c, sym] for s, (c, _, sym) in STRATUM_STYLE.items()}
    labels = {run: "Qwen3.5-2B (trained)", control: "Qwen3.5-2B, random init (untrained)",
              "expected": "Expected: Brownian motion on the reference tree"}
    data = {"names": d["names"], "strata": d["strata"], "modal": d["modal_layer"], "reps": reps, "tree": d["tree"],
            "style": style, "labels": labels, "left": run, "lims": _lims(d["reps"][run]["coords"])}
    opts = "".join(f'<option value="{k}">{labels[k]}</option>' for k in [control, "expected"] if k in reps)
    return f"""<div class="p3c" style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0">
<button id="p3-play" style="font:inherit;padding:3px 12px">▶ Play</button>
<input type="range" id="p3-layer" min="0" max="{len(reps[run]['coords']) - 1}" step="0.01" value="{d['modal_layer']}"
 style="flex:1;min-width:220px" aria-label="layer"><b id="p3-label" style="min-width:110px"></b>
<label>right panel <select id="p3-rep">{opts}</select></label>
<label>speed <select id="p3-speed"><option value="0.6">slow</option><option value="1.2" selected>normal</option>
<option value="2.5">fast</option></select></label>
<label><input type="checkbox" id="p3-tree" checked> tree overlay</label>
<label><input type="checkbox" id="p3-rot" checked> auto-rotate</label></div>
<div id="p3-plot" style="height:680px"></div>
<script>
(function () {{
  const D = {json.dumps(data, separators=(",", ":"))};
  const T = D.tree, N = T.parent.length, n = T.n, root = T.parent.indexOf(-1);
  const groups = Object.keys(D.style).map(s => ({{s, idx: D.strata.map((x, i) => x === s ? i : -1).filter(i => i >= 0)}}))
    .filter(g => g.idx.length);
  let right = '{control}', t = D.modal, playing = false, rotating = true, theta = 0.6, last = null, lastDraw = 0,
      dirty = true, syncing = false;
  const el = id => document.getElementById(id), plot = el('p3-plot');
  function coordsAt(r, tt) {{
    const C = D.reps[r].coords;
    if (C.length === 1) return C[0];
    const k = Math.min(Math.floor(tt), C.length - 2), f = tt - k, A = C[k], B = C[k + 1];
    return A.map((p, i) => [p[0] + (B[i][0] - p[0]) * f, p[1] + (B[i][1] - p[1]) * f, p[2] + (B[i][2] - p[2]) * f]);
  }}
  const limAt = tt => {{ const M = D.lims; if (M.length === 1) return M[0];
    const k = Math.min(Math.floor(tt), M.length - 2); return M[k] + (M[k + 1] - M[k]) * (tt - k); }};
  let LIM = limAt(D.modal);  // shared box half-width, interpolated between layers like the positions
  const outside = p => Math.abs(p[0]) > LIM || Math.abs(p[1]) > LIM || Math.abs(p[2]) > LIM;
  function treeLines(P) {{  // internal node = mean of descendant leaves (postorder: children before parents)
    const s = Array.from({{length: N}}, () => [0, 0, 0]), c = new Array(N).fill(0);
    for (let v = 0; v < n; v++) {{ s[v] = P[v].slice(); c[v] = 1; }}
    for (const v of T.postorder) if (v !== root) {{ const p = T.parent[v];
      s[p][0] += s[v][0]; s[p][1] += s[v][1]; s[p][2] += s[v][2]; c[p] += c[v]; }}
    const m = s.map((q, v) => [q[0] / c[v], q[1] / c[v], q[2] / c[v]]), x = [], y = [], z = [];
    for (let v = 0; v < N; v++) if (v !== root) {{ const p = T.parent[v];
      if (outside(m[p]) || outside(m[v])) continue;  // segments leaving the shared box are not drawn
      x.push(m[p][0], m[v][0], null); y.push(m[p][1], m[v][1], null); z.push(m[p][2], m[v][2], null); }}
    return [x, y, z];
  }}
  function title(r, P) {{
    const E = D.reps[r].evr, k = Math.round(Math.min(t, E.length - 1)), e = E[k], o = P.filter(outside).length;
    return '<b>' + D.labels[r] + '</b><br>' + (E.length === 1 ? 'one shape (no layers)' : 'layer ' + k) +
      ' · top-3 PCs explain ' + (100 * (e[0] + e[1] + e[2])).toFixed(1) + '%' +
      (o ? ' · ' + o + ' name' + (o > 1 ? 's' : '') + ' outside the box' : '');
  }}
  function panel(P, scene, show) {{
    const [x, y, z] = treeLines(P), q = (i, a) => outside(P[i]) ? null : P[i][a];
    const out = [{{type: 'scatter3d', mode: 'lines', x, y, z, scene, name: 'reference tree (centroids)', legendgroup: 'tree',
      showlegend: show, hoverinfo: 'skip', line: {{color: '#b9b8b2', width: 1.5}}, opacity: 0.5,
      visible: el('p3-tree').checked}}];
    for (const g of groups) out.push({{type: 'scatter3d', mode: 'markers', scene, name: g.s, legendgroup: g.s,
      showlegend: show, text: g.idx.map(i => D.names[i]), x: g.idx.map(i => q(i, 0)), y: g.idx.map(i => q(i, 1)),
      z: g.idx.map(i => q(i, 2)), hovertemplate: '<b>%{{text}}</b><br>' + g.s + '<extra></extra>',
      marker: {{color: D.style[g.s][0], symbol: D.style[g.s][1], size: 3.5, line: {{color: '#fcfcfb', width: 0.5}}}}}});
    return out;
  }}
  const both = () => [coordsAt(D.left, t), coordsAt(right, t)];
  const traces = ([PL, PR]) => panel(PL, 'scene', true).concat(panel(PR, 'scene2', false));
  const ax = name => ({{range: [-LIM, LIM], showticklabels: false, title: {{text: name, font: {{size: 12, color: '#52514e'}}}},
    backgroundcolor: '#fcfcfb', gridcolor: '#e1e0d9', zerolinecolor: '#c3c2b7'}});
  const sceneLayout = (x0, x1) => ({{domain: {{x: [x0, x1], y: [0.04, 0.9]}}, aspectmode: 'cube',
    xaxis: ax('axis 1'), yaxis: ax('axis 2'), zaxis: ax('axis 3')}});
  const eye = () => ({{x: 1.55 * Math.cos(theta), y: 1.55 * Math.sin(theta), z: 0.6}});
  const note = (x, text) => ({{x, y: 1, xref: 'paper', yref: 'paper', xanchor: 'center', yanchor: 'top', showarrow: false,
    text, font: {{size: 13, color: '#0b0b0b'}}}});
  const P0 = both();
  const layout = {{paper_bgcolor: '#fcfcfb', margin: {{l: 0, r: 0, t: 6, b: 0}}, uirevision: 'keep',
    font: {{family: 'system-ui, -apple-system, Segoe UI, sans-serif', color: '#0b0b0b'}},
    legend: {{orientation: 'h', yref: 'container', y: 0.005, yanchor: 'bottom', x: 0, itemsizing: 'constant', font: {{size: 11}}}},
    scene: {{...sceneLayout(0, 0.495), camera: {{eye: eye()}}}}, scene2: {{...sceneLayout(0.505, 1), camera: {{eye: eye()}}}},
    annotations: [note(0.25, title(D.left, P0[0])), note(0.75, title(right, P0[1]))]}};
  Plotly.newPlot(plot, traces(P0), layout, {{responsive: true, displaylogo: false}});
  function stopRotation() {{ if (rotating) {{ rotating = false; el('p3-rot').checked = false; }} }}
  function link(ev) {{  // mirror a dragged camera onto the other panel
    if (syncing) return;
    const k = Object.keys(ev).find(key => key === 'scene.camera' || key === 'scene2.camera');
    if (!k) return;
    stopRotation(); syncing = true;
    Plotly.relayout(plot, {{[k === 'scene.camera' ? 'scene2.camera' : 'scene.camera']: ev[k]}})
      .then(() => {{ syncing = false; }}, () => {{ syncing = false; }});
  }}
  plot.on('plotly_relayouting', link);
  plot.on('plotly_relayout', link);
  function draw() {{
    LIM = limAt(t);
    const P = both(), tr = traces(P);
    const upd = {{x: tr.map(a => a.x), y: tr.map(a => a.y), z: tr.map(a => a.z), visible: tr.map(a => a.visible ?? true)}};
    const lay = {{'annotations[0].text': title(D.left, P[0]), 'annotations[1].text': title(right, P[1])}};
    for (const sc of ['scene', 'scene2']) for (const a of ['xaxis', 'yaxis', 'zaxis']) lay[sc + '.' + a + '.range'] = [-LIM, LIM];
    if (rotating) {{ lay['scene.camera.eye'] = eye(); lay['scene2.camera.eye'] = eye(); }}
    syncing = true;
    Plotly.update(plot, upd, lay).then(() => {{ syncing = false; }}, () => {{ syncing = false; }});
    el('p3-label').textContent = 'layer ' + Math.round(t);
    dirty = false;
  }}
  let visible = true;
  new IntersectionObserver(e => {{ visible = e[0].isIntersecting; }}).observe(plot);
  function tick(ts) {{
    const dt = last === null ? 0 : Math.min((ts - last) / 1000, 1.0); last = ts;  // wall-clock speed, even at low frame rates
    const L = D.reps[D.left].coords.length - 1;
    if (playing) {{ t += dt * parseFloat(el('p3-speed').value); if (t > L) t = 0; el('p3-layer').value = t; dirty = true; }}
    if (rotating) {{ theta += dt * 0.35; dirty = true; }}
    if (dirty && visible && !syncing && ts - lastDraw > 40) {{ lastDraw = ts; draw(); }}
    requestAnimationFrame(tick);
  }}
  el('p3-play').onclick = () => {{ playing = !playing; el('p3-play').textContent = playing ? '❚❚ Pause' : '▶ Play'; }};
  el('p3-layer').oninput = e => {{ t = parseFloat(e.target.value); dirty = true; }};
  el('p3-rep').onchange = e => {{ right = e.target.value; dirty = true; }};
  el('p3-tree').onchange = () => {{ dirty = true; }};
  el('p3-rot').onchange = e => {{ rotating = e.target.checked; dirty = true; }};
  requestAnimationFrame(tick);
}})();
</script>"""


def fig_pca3d(run: str, tree, control: str = "qwen35_2b_random") -> dict | None:
    """Animated, interactive side-by-side 3-D view of the top-3 principal components at every layer: the trained
    model next to its random-init twin (custom player for the site), plus a static snapshot at the confirmatory
    layer (PNG) and an animated GIF (README)."""
    d = _load(RESULTS_DIR / run / "viz" / "pca3d.json")
    checks = _load(RESULTS_DIR / run / "stage1a" / "checks.json")
    if not d or control not in d["reps"]:
        return None
    L = d["modal_layer"]
    lim = _lims(d["reps"][run]["coords"])[L]
    snap = _pair_fig([(np.array(d["reps"][q]["coords"][L]), pca3d_title(d, q, L, lim)) for q in (run, control)],
                     d["names"], d["strata"], tree, lim)
    k = sum("tree-consistent" in c["reading"] for c in checks) if checks else 0
    first = next((c["layer"] for c in checks or [] if "tree-consistent" in c["reading"]), None)
    r = rec("pca3d", "Top-3 principal components in 3-D, animated through all layers: trained vs random init", "mixed",
            f"trained: H1a checks pass in {k}/{len(checks or [])} layers"
            + (f", already at near-lexical layer {first}, so they do not discriminate" if first is not None and first <= 2 else "")
            + "; by eye: class arms, no nested branching",
            "Left: the trained model. Right: the same architecture with random weights (the untrained control), at the "
            "same layer. Each point is one species at the last token of its name (mean over 8 prompts), projected on that "
            "layer's top three principal components. Consecutive layers are rotated onto each other (orthogonal "
            "Procrustes, display only) so the clouds move smoothly instead of flipping axes, and each layer is scaled so "
            "the median species sits at radius 1. Both panels share one box per layer, sized to hold every trained-model "
            "species at that layer; species of the untrained network that fall outside it are not drawn and are counted in its "
            "title. Gray lines draw the reference tree with each internal node at the mean of its descendant "
            "species. One clock drives both panels and their cameras are linked: press Play, drag either panel to rotate "
            "both, scroll to zoom, hover for names, click legend entries to toggle a stratum in both, or switch the right "
            "panel to the expected shape of a true tree code (Brownian motion on the reference tree). The trained model "
            "separates coarse groups into arms; the untrained network's top components are dominated by a few outlier "
            "names, with the remaining species in one mixed cloud. The registered checks (trained model) find class "
            "separation and some order-level clustering; the nested branching of the expected tree is not visible in "
            "three components.", snap, (1300, 680))
    r["html"] = _player_html(d, run, control)
    r["pca3d"] = d
    r["control"] = control
    r["gifs"] = {"pca3d": lambda path: export_pca3d_gif(r, tree, path)}  # README animation (export_pngs)
    return r


def pca3d_title(d: dict, rep: str, layer: float, lim: float, P=None) -> str:
    """Panel title: model, layer, variance of the top-3 PCs, and how many species fall outside the shared box."""
    names = {"qwen35_2b": "Qwen3.5-2B (trained)", "qwen35_2b_random": "Qwen3.5-2B, random init"}
    lay = int(round(layer))
    P = np.array(d["reps"][rep]["coords"][lay]) if P is None else P
    o = n_outside(P, lim)
    return (f"<b>{names.get(rep, rep)}</b> · layer {lay} of {len(d['reps'][rep]['coords']) - 1} · top-3 PCs "
            f"{100 * sum(d['reps'][rep]['evr'][lay]):.1f}%" + (f" · {o} outside the box" if o else ""))


def export_pca3d_gif(r: dict, tree, path: Path, steps: int = 2, size=(1080, 500)) -> Path:
    """README animation: trained (left) and random-init (right) side by side, sweeping layers 0 -> last
    (``steps`` interpolated frames per layer) while both cameras turn together (assembled by :func:`_save_gif`)."""
    d, run, ctl = r["pca3d"], "qwen35_2b", r["control"]
    C = {k: np.array(d["reps"][k]["coords"]) for k in (run, ctl)}
    lims = _lims(C[run])  # one box per layer, shared by both panels
    nL = len(C[run])
    ts = np.linspace(0, nL - 1, steps * (nL - 1) + 1)
    figs = []
    for j, tt in enumerate(ts):
        k = min(int(tt), nL - 2)
        th = 0.6 + 2 * np.pi * j / len(ts)
        Ps = {q: C[q][k] + (C[q][k + 1] - C[q][k]) * (tt - k) for q in (run, ctl)}
        lim = _lim_at(lims, tt)
        panels = [(Ps[q], pca3d_title(d, q, tt, lim, Ps[q])) for q in (run, ctl)]
        f = _pair_fig(panels, d["names"], d["strata"], tree, lim, eye=(1.55 * np.cos(th), 1.55 * np.sin(th), 0.6))
        f.add_annotation(text=f"<span style='color:{BLUE}'>■ vertebrates</span>   <span style='color:{ORANGE}'>■ "
                              f"arthropods</span>   <span style='color:{AQUA}'>■ other invertebrates</span>   <span "
                              f"style='color:{GRAY}'>■ other</span>   gray lines: reference tree",
                         x=0.5, y=0.0, xref="paper", yref="paper", xanchor="center", yanchor="bottom", showarrow=False,
                         font={"size": 12, "color": INK2})
        figs.append(f.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 6, "b": 26}))
    return _save_gif(figs, path, size, duration=180)


def gif_palette():
    """Fixed 'P'-mode palette for the GIFs: the exact chart colours, each blended 30/55/80% toward the surface (for
    anti-aliased edges), plus a 40-step gray ramp for text and grid. Adaptive (median-cut) palettes spend their
    entries on the many gray pixels and shifted aqua to a grayish teal, so series colours did not match the legend.
    Pillow: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.quantize"""
    from PIL import Image

    rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
    surf = rgb(SURFACE)
    keep = [SURFACE, INK, INK2, MUTED, GRID, AXIS, LIGHTGRAY, TREE_GRAY, OTHER_ORDERS[1], BLUE, ORANGE, AQUA, GRAY]
    cols = [rgb(h) + (surf - rgb(h)) * t for h in keep for t in (0, 0.3, 0.55, 0.8)]
    cols += [np.full(3, v) for v in np.linspace(11, 252, 40)]
    cols += [surf] * (256 - len(cols))  # pad with the background, never with black
    img = Image.new("P", (1, 1))
    img.putpalette(np.clip(np.rint(cols), 0, 255).astype(np.uint8).ravel().tolist())
    return img


def _save_gif(figs: list[go.Figure], path: Path, size: tuple[int, int], duration: int) -> Path:
    """Render Plotly frames with kaleido in one batch (https://plotly.com/python/static-image-export/) and assemble
    a looping GIF with Pillow (https://pillow.readthedocs.io/), every frame mapped to :func:`gif_palette`."""
    import tempfile

    import plotly.io as pio
    from PIL import Image

    pal = gif_palette()
    with tempfile.TemporaryDirectory() as tmp:
        files = [Path(tmp) / f"f{j:03d}.png" for j in range(len(figs))]
        pio.write_images(figs, files, width=size[0], height=size[1], scale=1)
        frames = [Image.open(f).convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE) for f in files]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    log.info("animated GIF (%d frames, %.1f MB) written to %s", len(figs), path.stat().st_size / 1e6, path)
    return path


# ======================================================================================
# Tree order inside groups, in 3-D, with the reference tree fitted into each cloud (post hoc)
# ======================================================================================

OTHER_ORDERS = ("other orders", "#bdbcb6", "circle")  # recessive gray bucket, never a 4th hue (dataviz rule)
ORDER_HUES = [(BLUE, "circle"), (ORANGE, "diamond"), (AQUA, "square")]  # validated 3-hue set; symbol = redundancy
GLYPH = {"circle": "●", "diamond": "◆", "square": "■"}


def _within_columns(w3: dict) -> list[tuple[str, str]]:
    """(key, label) of the three columns: the expected tree shape, the model, its untrained twin."""
    return [("expected", "Expected: a Brownian reference draw"), (w3["main"], f"Qwen3.5-2B, layer {w3['layer']}"),
            (w3["control"], "Qwen3.5-2B, random weights")]


def _order_style(orders: np.ndarray) -> tuple[dict, np.ndarray]:
    """Colour key for one group: its 3 largest orders get the validated hues (ties broken by name), every other
    order (and a blank order) folds into gray 'other orders'. Returns (label -> (colour, symbol), per-species label)."""
    names, counts = np.unique(orders[orders != ""], return_counts=True)
    top = [n for _, n in sorted(zip(-counts, names))][:3]
    style = dict(zip(top, ORDER_HUES)) | {OTHER_ORDERS[0]: OTHER_ORDERS[1:]}
    return style, np.where(np.isin(orders, top), orders, OTHER_ORDERS[0])


def _within_group(g: dict, df, tree) -> dict:
    """Per-group drawing inputs: the group's reference subtree (leaves in the order of g['idx']), names, colour
    labels and hover detail (order · family)."""
    idx = np.array(g["idx"])
    d = df.iloc[idx]
    style, labels = _order_style(d["order"].to_numpy())
    detail = [f"{o or '(no order)'} · {f or '(no family)'}" for o, f in zip(d["order"], d["family"])]
    return {"sub": tree.subtree(list(idx)), "names": d["common_name"].tolist(), "labels": labels, "style": style,
            "detail": detail}


def _within_box(C: np.ndarray) -> float:
    """Half-width of a panel's cube: holds every species unless a few lie beyond 1.5x the 95th-percentile extent
    (only the random-weight mammals have such outliers); those are clipped and counted in the title."""
    m = np.abs(C).max(1)
    return round(1.05 * float(min(m.max(), 1.5 * np.quantile(m, 0.95))), 3)


def _within_traces(C: np.ndarray, gd: dict, scene: str, showlegend: bool,
                   lim: float | None = None) -> tuple[list, float, int]:
    """Traces for one panel (3-D coords ``C`` of the group's species), its box half-width and the number of species
    outside the box: the reference subtree fitted into the points (phylomorphospace,
    :func:`phylogeo.viz.pca.fit_tree_positions`) and the species. Branches whose whole clade lies inside one
    highlighted order wear that order's colour; deeper branches are gray. Lines are opaque: semi-transparent 3-D
    lines break marker depth sorting in WebGL (markers render cut)."""
    from phylogeo.viz.pca import fit_tree_positions

    sub, labels = gd["sub"], gd["labels"]
    lim = _within_box(C) if lim is None else lim
    pos = _clip(fit_tree_positions(sub, C), lim)  # fitted on all points; segments leaving the box are not drawn
    lines = [_tree_lines(sub, pos, scene, vs, color=c, name="reference tree, fitted", showlegend=showlegend and
                         c == LIGHTGRAY, width=1.5 if c == LIGHTGRAY else 2.5, opacity=1.0)
             for c, vs in _edge_colours(gd).items()]
    pts = _point_traces(_clip(C, lim), gd["names"], labels, gd["style"], scene, showlegend, size=4.5,
                        detail=gd["detail"])
    return lines + pts, lim, n_outside(C, lim)


def _edge_colours(gd: dict) -> dict[str, list[int]]:
    """Tree edges grouped by colour: an edge whose whole clade lies inside one highlighted order wears that order's
    colour, every other edge is light gray."""
    sub, labels, out = gd["sub"], gd["labels"], {}
    for v in range(sub.N):
        if v == sub.root:
            continue
        labs = set(labels[sub.members[v]])
        one = labs.pop() if len(labs) == 1 else OTHER_ORDERS[0]
        out.setdefault(gd["style"][one][0] if one != OTHER_ORDERS[0] else LIGHTGRAY, []).append(v)
    return out


def _within_title(label: str, s: dict, dims: int, n_out: int = 0) -> str:
    """Panel title: what it is, triplet accuracy of the 3-D picture and of all ``dims`` dimensions, the tree fit
    (from ``s``), and how many species fall outside the box (never silently dropped)."""
    return (f"<b>{label}</b><br>triplet accuracy: 3-D {s['ta_3d']:.3f} · all {dims} dims "
            f"{s['ta_full']:.3f}<br>tree fit {s['fit_ratio']:.2f} (1 = random species)"
            + (f" · {n_out} outside the box" if n_out else ""))


def _within_panel(w3: dict, g: dict, key: str, label: str, t: float | None) -> tuple:
    """(coords, stats, label, box) of one panel. ``t`` = None: the static view at the H1 layer. ``t`` = a layer
    position: the layer sweep, with positions interpolated linearly between the two neighbouring (Procrustes-
    aligned) layers, the box interpolated the same way, and the numbers of the nearest layer. The expected tree
    shape has no layers and is always static."""
    p = g["panels"][key]
    if t is None or "layers" not in p:
        C = np.array(p["coords"])
        return C, p, label, _within_box(C)
    L = p["layers"]
    Cs = [np.array(c) for c in L["coords"]]
    k = min(int(t), len(Cs) - 2)
    C = Cs[k] + (Cs[k + 1] - Cs[k]) * (t - k)
    near = int(round(t))
    s = {m: L[m][near] for m in ("ta_3d", "ta_full", "fit_ratio")}
    name = label.split(",")[0] + (", random weights" if key == w3["control"] else "")
    tag = f"{name} · layer {near} of {len(Cs) - 1}" + (" (H1 layer)" if near == w3["layer"] else "")
    return C, s, tag, _lim_at([_within_box(c) for c in Cs], t)


def _within_scene(dx, dy, lim: float, eye) -> dict:
    """One 3-D scene: equal axes in a cube of half-width ``lim`` (no ticks; PCs are unitless), recessive grid."""
    ax = {"range": [-lim, lim], "showticklabels": False, "backgroundcolor": SURFACE, "gridcolor": GRID,
          "zerolinecolor": AXIS, "showspikes": False}
    font = {"size": 11, "color": MUTED}
    return {"domain": {"x": dx, "y": dy}, "aspectmode": "cube", "camera": {"eye": dict(zip("xyz", eye))},
            **{f"{a}axis": {**ax, "title": {"text": f"PC{k}", "font": font}} for k, a in enumerate("xyz", start=1)}}


def _within_key(name: str, g: dict, style: dict) -> str:
    """Row header: group, decision nodes, and the colour key (coloured glyph, label in ink)."""
    key = "   ".join(f"<span style='color:{c}'>{GLYPH[s]}</span> {o}" for o, (c, s) in style.items())
    return f"<b>{name}</b> ({g['n_nodes']} decision nodes)   {key}   · coloured branches: a clade inside one order"


def _within_grid(w3: dict, groups: list, gds: dict, eye=(1.7, 1.25, 0.8), row_px: int = 400, width: int = 1200,
                 t: float | None = None):
    """Static grid (PNG / GIF frames): one row per group, columns = expected tree, model, random weights. ``t``:
    layer position for the layer-sweep GIF (None = the H1 layer)."""
    fig, cols, R = go.Figure(), _within_columns(w3), len(groups)
    H = R * row_px
    head, title = 26 / H, 54 / H  # header line and 3-line panel titles, as fractions of the figure height
    _style(fig, H, legend=False).update_layout(margin={"l": 0, "r": 0, "t": 4, "b": 0}, width=width)
    for i, (name, g) in enumerate(groups):
        y1 = 1 - i / R
        fig.add_annotation(text=_within_key(name, g, gds[name]["style"]), x=0.005, y=y1, xref="paper", yref="paper",
                           showarrow=False, xanchor="left", yanchor="top", font={"size": 13, "color": INK})
        for j, (key, label) in enumerate(cols):
            n = i * len(cols) + j
            sc = "scene" if n == 0 else f"scene{n + 1}"
            C, s, tag, box = _within_panel(w3, g, key, label, t)
            tr, lim, n_out = _within_traces(C, gds[name], sc, showlegend=False, lim=box)
            fig.add_traces(tr)
            fig.update_layout({sc: _within_scene([j / 3 + 0.004, (j + 1) / 3 - 0.004],
                                                 [y1 - 1 / R + 24 / H, y1 - head - title], lim, eye)})
            fig.add_annotation(text=_within_title(tag, s, w3["hidden_size"], n_out), x=(j + 0.5) / 3, y=y1 - head,
                               xref="paper",
                               yref="paper", showarrow=False, xanchor="center", yanchor="top",
                               font={"size": 11.5, "color": INK2})
    return fig


def _within_dropdown(w3: dict, groups: list, gds: dict, eye=(1.5, 1.1, 0.7)) -> go.Figure:
    """Site version: three linked-size scenes and a dropdown that switches the group (few WebGL contexts)."""
    fig, cols = go.Figure(), _within_columns(w3)
    _legend_below(_style(fig, 640), bottom=60, itemsizing="constant").update_layout(margin={"l": 0, "r": 0, "t": 120})
    spans, layouts = [], []
    for i, (name, g) in enumerate(groups):
        n0, lay, titles = len(fig.data), {}, []
        for j, (key, label) in enumerate(cols):
            sc = "scene" if j == 0 else f"scene{j + 1}"
            C, s, tag, box = _within_panel(w3, g, key, label, None)
            tr, lim, n_out = _within_traces(C, gds[name], sc, showlegend=(j == 0), lim=box)
            for t in tr:
                t.visible = i == 0
            fig.add_traces(tr)
            for a in "xyz":
                lay[f"{sc}.{a}axis.range"] = [-lim, lim]
            titles.append(_within_title(tag, s, w3["hidden_size"], n_out))
            if i == 0:
                fig.update_layout({sc: _within_scene([j / 3 + 0.004, (j + 1) / 3 - 0.004], [0.0, 0.86], lim, eye)})
        lay["annotations"] = [{"text": t, "x": (j + 0.5) / 3, "y": 1.0, "xref": "paper", "yref": "paper",
                               "showarrow": False, "xanchor": "center", "yanchor": "top",
                               "font": {"size": 12, "color": INK2}} for j, t in enumerate(titles)]
        spans.append((n0, len(fig.data)))
        layouts.append(lay)
    buttons = [{"label": f"{name} ({g['n_nodes']} decision nodes)", "method": "update",
                "args": [{"visible": [a <= k < b for k in range(len(fig.data))]}, layouts[i]]}
               for i, ((name, g), (a, b)) in enumerate(zip(groups, spans))]
    fig.update_layout(updatemenus=[{"type": "dropdown", "buttons": buttons, "x": 0.0, "y": 1.0, "xanchor": "left",
                                    "yanchor": "bottom", "pad": {"b": 12}, "showactive": True}],
                      annotations=layouts[0]["annotations"])
    return fig


def export_within3d_gif(w3: dict, groups: list, gds: dict, path: Path, n_frames: int = 24,
                        size=(1080, 1360)) -> Path:
    """README animation: the static grid with all scene cameras turning together, one full turn (24 frames keep
    the file a few MB for GitHub)."""
    R = len(groups)
    figs = [_within_grid(w3, groups, gds, eye=(2.0 * np.cos(th), 2.0 * np.sin(th), 0.8), row_px=size[1] // R,
                         width=size[0]) for th in 0.6 + 2 * np.pi * np.arange(n_frames) / n_frames]
    return _save_gif(figs, path, (size[0], (size[1] // R) * R), duration=200)


def export_within3d_layers_gif(w3: dict, groups: list, gds: dict, path: Path, steps: int = 2,
                               size=(1080, 1360)) -> Path:
    """README animation through the layers: both networks sweep layer 0 -> last (``steps`` interpolated frames per
    layer) while every camera makes one full turn; the expected tree shape stays fixed (it has no layers)."""
    R = len(groups)
    nL = len(groups[0][1]["panels"][w3["main"]]["layers"]["coords"])
    ts = np.linspace(0, nL - 1, steps * (nL - 1) + 1)
    figs = [_within_grid(w3, groups, gds, eye=(2.0 * np.cos(th), 2.0 * np.sin(th), 0.8), row_px=size[1] // R,
                         width=size[0], t=float(t)) for t, th in zip(ts, 0.6 + 2 * np.pi * np.arange(len(ts)) / len(ts))]
    return _save_gif(figs, path, (size[0], (size[1] // R) * R), duration=180)


def fig_within3d(run: str, tree, df) -> dict | None:
    """3-D views of tree order inside the large groups: a Brownian reference draw, the model, and its random-weight
    twin, each with the reference subtree fitted into the cloud (post hoc; data from ``within3d.json``)."""
    w3 = _load(RESULTS_DIR / run / "viz" / "within3d.json")
    if not w3 or not w3["groups"]:
        return None
    groups = sorted(w3["groups"].items(), key=lambda kv: -kv[1]["n_nodes"])
    gds = {name: _within_group(g, df, tree) for name, g in groups}
    clear = {s: [name for name, g in groups if g[f"diff_ci95_{s}"][0] > 0] for s in ("3d", "full")}
    kind = ("positive" if len(clear["full"]) == len(groups) else
            "negative" if not clear["3d"] and not clear["full"] else "mixed")
    verdict = (f"beats random weights in {len(clear['3d'])}/{len(groups)} groups in 3-D"
               + (f" ({', '.join(clear['3d'])})" if clear["3d"] else "")
               + f", in {len(clear['full'])}/{len(groups)} in all dimensions (per-group CIs)")
    caption = (f"Post hoc, descriptive. Groups: every group with at least {w3['min_nodes']} decision nodes in the H1 "
               f"test (one row each in the static view). Left: a Brownian reference draw (Brownian motion on the reference "
               f"tree), middle: "
               f"Qwen3.5-2B at layer {w3['layer']} (the H1 layer), right: the same network with random weights. Every "
               "panel goes through one pipeline: species vectors scaled to unit length (the geometry the cosine-based "
               "H1 test scores), centred on the group, top three principal components. Colours: the group's three "
               "largest orders; gray: all other orders. Lines: the reference tree fitted into the points "
               "(phylomorphospace: internal nodes where the total squared branch length is smallest, i.e. Brownian-"
               "motion ancestral states); branches inside one coloured order wear its colour. Titles give the "
               f"triplet accuracy of the 3-D picture and of all {w3['hidden_size']} dimensions (chance 1/3), and the "
               f"tree fit: squared length of the fitted tree over its mean across {w3['n_fit_perm']} shuffles of the "
               "species (1 = no better than a random assignment). The tree fit uses every species, including names that "
               "share a head noun (\"… beetle\"), so shared words can lower it even in the random-weight network; the "
               "triplet accuracies exclude such names, as H1 does. Three dimensions keep only part of the order, even "
               "for the Brownian reference draw. Drag to rotate, hover for names, pick a group above.")
    r = rec("within3d", "Tree order inside groups, in 3-D: trained vs random weights, with the tree fitted in", kind,
            verdict, caption, _within_dropdown(w3, groups, gds), (1200, 400 * len(groups)))
    r["png_fig"] = _within_grid(w3, groups, gds)
    r["gifs"] = {"within3d": lambda path: export_within3d_gif(w3, groups, gds, path),
                 "within3d_layers": lambda path: export_within3d_layers_gif(w3, groups, gds, path)}
    r["gallery_img"] = "within3d_layers.gif"  # the README gallery shows the layer sweep instead of the PNG
    r["within3d"], r["groups"], r["clear"] = w3, groups, clear
    return r


def within3d_md(r: dict, png_dir: str = "reports/figures/viz") -> str:
    """README block next to the H1 highlight: two animations linked to the interactive card (a sweep through all
    layers, then the H1 layer rotating) and a generated table of the per-group numbers, with trained-minus-random
    CIs."""
    w3, groups = r["within3d"], r["groups"]
    main, ctl, D = w3["main"], w3["control"], w3["hidden_size"]
    nL = len(groups[0][1]["panels"][main]["layers"]["coords"])

    def gif(name, note):
        return (f'<p align="center"><a href="{SITE_URL}#fig-within3d"><img src="{png_dir}/{name}.gif" '
                f'alt="{r["title"]}: {note}" width="100%"></a><br><b>{note}</b></p>\n\n')

    def cell(g, k, ci):
        p = g["panels"]
        return f"{p[main][k]:.3f} / {p[ctl][k]:.3f} [{g[ci][0]:+.3f}, {g[ci][1]:+.3f}]"

    rows = [f"| {name} ({g['n_nodes']}) | {cell(g, 'ta_3d', 'diff_ci95_3d')} | {cell(g, 'ta_full', 'diff_ci95_full')} | "
            + " / ".join(f"{g['panels'][k]['fit_ratio']:.2f}" for k in ("expected", main, ctl)) + " |"
            for name, g in groups]
    return (f'<p align="center"><b>{r["title"]}</b> (animated here; interactive on the site)<br>'
            f'{KIND_LABEL[r["kind"]]}: {r["verdict"]}</p>\n\n'
            + gif("within3d_layers", f"Through the layers: both networks sweep layers 0 → {nL - 1} (the Brownian "
                                     "reference has no layers); each title shows that layer's numbers")
            + gif("within3d", f"At the H1 layer ({w3['layer']}), rotating; the table below is at this layer")
            + "*Generated by `python -m phylogeo.run --phases viz_data,site` from `results/`; post hoc, descriptive.* "
            "In the layer sweep, each layer has its own projection; consecutive layers are rotated onto each other "
            "(orthogonal Procrustes, display only) and positions are interpolated between layers, so the clouds move "
            "smoothly instead of flipping axes. "
            + r["caption"].replace("Post hoc, descriptive. ", "").replace(" Drag to rotate, hover for names, pick a "
                                                                            "group above.", "")
            + " Internal-node placement follows squared-change parsimony ([Maddison 1991](https://search.r-project.org/"
            "CRAN/refmans/castor/html/asr_squared_change_parsimony.html)) as in phylomorphospace plots ([Sidlauskas "
            "2008](https://doi.org/10.1111/j.1558-5646.2008.00519.x); phytools `phylomorphospace3d`, [Revell 2012]"
            "(https://doi.org/10.1111/j.2041-210X.2011.00169.x)).\n\n"
            f"| group (decision nodes) | triplet accuracy, 3-D view: trained / random weights [difference, 95% CI] | "
            f"all {D} dimensions: trained / random weights [difference, 95% CI] | tree fit, lower = better: Brownian "
            "reference / trained / random weights |\n|---|---|---|---|\n" + "\n".join(rows)
            + f"\n\nCIs: paired family-block bootstrap ({w3['n_boot']} draws) of trained minus random weights, on the "
            f"H1 triplets of each group. The difference excludes zero in {len(r['clear']['3d'])} of {len(groups)} groups "
            f"in the 3-D view and in {len(r['clear']['full'])} of {len(groups)} in all {D} dimensions. Group by group, "
            "the picture therefore shows the separation more sharply than the full space does; the confirmatory "
            "evidence is the registered H1 test, which pools all groups (key-results table below).\n")


def fig_group_view(run: str, tree, df, group: str = "birds") -> dict | None:
    """Static side-by-side view of one group at the H1 layer: the Brownian reference draw (ground truth), the model and
    its random-weight twin, each seen from the camera angle where its fitted tree looks most tree-like
    (:func:`phylogeo.viz.pca.best_view`, the same rule for every panel), drawn flat (orthographic) with names."""
    from phylogeo.viz.pca import fit_tree_positions

    w3 = _load(RESULTS_DIR / run / "viz" / "within3d.json")
    if not w3 or group not in w3["groups"] or "view" not in w3["groups"][group]["panels"]["expected"]:
        return None
    g, cols = w3["groups"][group], _within_columns(w3)
    gd = _within_group(g, df, tree)
    titles = [f"<b>{'Ground truth: a Brownian reference draw' if key == 'expected' else label}</b><br>triplet accuracy: "
              f"this view {g['panels'][key]['view']['ta_2d']:.3f} · 3-D {g['panels'][key]['ta_3d']:.3f} · all "
              f"{w3['hidden_size']} dims {g['panels'][key]['ta_full']:.3f}<br>tree fit: this view "
              f"{g['panels'][key]['view']['fit_ratio_2d']:.2f} · 3-D {g['panels'][key]['fit_ratio']:.2f} "
              "(1 = random species)" for key, label in cols]
    fig = make_subplots(1, 3, subplot_titles=titles, horizontal_spacing=0.015)
    labels, names, detail = gd["labels"], np.array(gd["names"]), np.array(gd["detail"])
    for j, (key, _) in enumerate(cols, start=1):
        p = g["panels"][key]
        C, B = np.array(p["coords"]), np.array(p["view"]["basis"])
        P, pos = C @ B, fit_tree_positions(gd["sub"], C) @ B  # the fit is linear: projecting it = fitting in the view
        for c, vs in _edge_colours(gd).items():
            xs, ys = _segments(gd["sub"], pos, vs)
            fig.add_scatter(x=xs, y=ys, mode="lines", line={"color": c, "width": 1.2 if c == LIGHTGRAY else 2},
                            name="reference tree, fitted", legendgroup="tree", showlegend=j == 1 and c == LIGHTGRAY,
                            hoverinfo="skip", row=1, col=j)
        shown = label_mask(P, names, np.array([list(gd["style"]).index(s) for s in labels]))  # coloured orders first
        for s, (color, sym) in gd["style"].items():
            m = labels == s
            fig.add_scatter(x=P[m, 0], y=P[m, 1], mode="markers+text", text=np.where(shown[m], names[m], ""),
                            textposition="top center", textfont={"size": 9, "color": INK2},
                            customdata=np.c_[names[m], detail[m]], name=s, legendgroup=s, showlegend=j == 1,
                            marker={"color": color, "symbol": sym, "size": 9, "line": {"color": SURFACE, "width": 1}},
                            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>", row=1, col=j)
        pad = 0.14 * float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1])))  # room for names at the edges
        fig.update_xaxes(range=[P[:, 0].min() - pad, P[:, 0].max() + pad], row=1, col=j)
        fig.update_yaxes(range=[P[:, 1].min() - pad / 2, P[:, 1].max() + pad], scaleanchor="x" if j == 1 else f"x{j}",
                         scaleratio=1, row=1, col=j)  # equal aspect: distances in the picture are true distances
    _legend_below(_style(fig, 780), bottom=70, itemsizing="constant").update_layout(margin={"l": 10, "r": 10, "t": 95})
    for a in fig.layout.annotations:
        a.font = {"size": 12, "color": INK2}
    for upd in (fig.update_xaxes, fig.update_yaxes):  # positions are unitless projections: no ticks or grid
        upd(showticklabels=False, showgrid=False, zeroline=False, ticks="", mirror=True, showline=True, linecolor=GRID)
    from phylogeo.prompts import TEMPLATES

    v = {k: g["panels"][k]["view"]["fit_ratio_2d"] for k, _ in cols}
    r = rec(f"view_{group.replace(' ', '_')}", f"{group.capitalize()} at layer {w3['layer']}. "
            "Each panel shows the optimized view for its condition (Brownian reference, trained or random weights).", "example",
            f"tree fit in the best view: Brownian reference {v['expected']:.2f} · trained {v[w3['main']]:.2f} · random "
            f"weights {v[w3['control']]:.2f} (1 = random species)",
            group_view_caption(w3, g, group, len(df), len(TEMPLATES)), fig, (1650, 800))
    r["in_gallery"] = False  # shown at the top of the README instead
    return r


def group_view_caption(w3: dict, g: dict, group: str, n_species: int, n_prompts: int) -> str:
    """Caption of :func:`fig_group_view`: every step from activations to the flat picture, so the figure can be read
    without the code. Counts come from the within3d record (``g``, ``w3``), the species table and the prompt list.
    Steps: readout and per-prompt centring (:func:`phylogeo.geometry.species_vectors`), unit length and group PCA
    (:func:`phylogeo.viz.viz_data.unit_group_pca3`), the angle (:func:`phylogeo.viz.pca.best_view`) and the fitted
    tree (:func:`phylogeo.viz.pca.fit_tree_positions`)."""
    D, L, e = w3["hidden_size"], w3["layer"], g["panels"]["expected"]
    n_blocks = len(g["panels"][w3["main"]]["layers"]["coords"]) - 1  # layer 0 is the embedding output
    return (f"The 3-D within-group view of {group} ({len(g['idx'])} species, {g['n_nodes']} decision nodes), seen from "
            f"one camera angle per panel and drawn flat. Pipeline: Qwen3.5-2B residual stream after layer {L} (the "
            f"output of block {L} of {n_blocks}; the H1 layer), at the last token of each species name in {n_prompts} "
            f"prompts. Within each prompt the mean over all {n_species} species is subtracted, then the prompts are "
            f"averaged. Each vector is scaled to unit length, centred on the {group} and projected onto the {group}' "
            "top 3 principal components. Angle rule, identical for every panel: of 2,000 evenly spread viewing "
            "directions, the one where the fitted reference tree is shortest relative to a random assignment of the "
            "species (exact expectation, no shuffles). The picture is therefore not the PC1–PC2 plane but the most "
            "tree-like view of each 3-D cloud. Random weights get their best angle too, so the comparison is not "
            "cherry-picked, but it is a display choice, not a test: the 'this view' numbers are optimistic, and the "
            "3-D and all-dimension numbers are the fair ones. Lines: the reference tree fitted into the points, as in "
            "phylomorphospace plots (leaves at the species, ancestor nodes where the total squared edge length is "
            "smallest), coloured inside one order; branch points are fitted positions, not species. Left: an ideal "
            f"tree code, one draw of Brownian motion along the reference tree in {D} dimensions with unit edge lengths "
            "(topology only), through the same unit-length, centring, PCA and angle steps (it has no prompts). Even "
            f"this code keeps only part of the tree after projection: triplet accuracy {e['ta_full']:.3f} in all {D} "
            f"dimensions, {e['ta_3d']:.3f} in 3-D and {e['view']['ta_2d']:.3f} in this view. Middle: Qwen3.5-2B; "
            "right: the same architecture with freshly initialized weights, same tokenizer and pipeline. Colours: the "
            "three largest orders, gray: other orders. Names are shown where they fit (hover on the site for every "
            "name). Titles: triplet accuracy (chance 1/3) and tree fit (1 = random species) in this view, in 3-D and "
            f"(accuracy) in all {D} dimensions.")


def label_mask(P: np.ndarray, names, priority: np.ndarray, size_px: float = 560, char_px: float = 5.4,
               line_px: float = 13) -> np.ndarray:
    """Which points get a visible name in a static 2-D panel: greedily, in ``priority`` order (lower first), a
    name is shown only if its text box (about ``char_px`` per character by ``line_px``, centred above the point)
    overlaps no name already shown. Coordinates are mapped to pixels with one scale for both axes (equal aspect,
    the panel's longer data side = ``size_px``). Unlabelled points keep their hover name on the site."""
    span = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), 1e-9))
    xy = (P - P.min(0)) / span * size_px
    boxes, keep = [], np.zeros(len(P), bool)
    for i in np.argsort(priority, kind="stable"):
        w = char_px * len(names[i])
        x0, x1, y0, y1 = xy[i, 0] - w / 2, xy[i, 0] + w / 2, xy[i, 1] + 3, xy[i, 1] + 3 + line_px
        if all(x1 < a or x0 > b or y1 < c or y0 > d for a, b, c, d in boxes):
            boxes.append((x0, x1, y0, y1))
            keep[i] = True
    return keep


def group_view_md(r: dict, png_dir: str = "reports/figures/viz") -> str:
    """README block at the top of the visual summary: the static best-angle view, linked to its site card."""
    return (f'<p align="center"><a href="{SITE_URL}#fig-{r["key"]}"><img src="{png_dir}/{r["key"]}.png" '
            f'alt="{r["title"]}" width="100%"></a><br><b>{r["title"]}</b><br>{KIND_LABEL[r["kind"]]}: '
            f'{r["verdict"]}</p>\n\n*Generated by `python -m phylogeo.run --phases viz_data,site` from `results/`; '
            f"post hoc, descriptive.* {r['caption']}\n")


def build_figures(run: str = "qwen35_2b") -> list[dict]:
    """All figure records whose inputs exist, in reading order (ground truth first)."""
    from phylogeo.phases import load_species

    df, tree = load_species()
    builders = [lambda: fig_pca3d(run, tree), lambda: fig_taxonomy(df), lambda: fig_distance_heatmaps(run, df),
                lambda: fig_expected_pca(run),
                lambda: fig_h1_forest(run), lambda: fig_group_view(run, tree, df), lambda: fig_within3d(run, tree, df),
                lambda: fig_layer_profile(run),
                lambda: fig_per_stratum(run),
                lambda: fig_depth_profile(run), lambda: fig_neighbor_levels(run), lambda: fig_neighbor_examples(run),
                lambda: fig_secondary(run), lambda: fig_conflict(run), lambda: fig_probe(run), lambda: fig_treeness(run),
                lambda: fig_causal(run), lambda: fig_habitat(run), lambda: fig_scale(run), lambda: fig_sensitivity(run)]
    out = []
    for b in builders:
        r = b()
        if r is not None:
            out.append(r)
    return out


SITE_URL = "https://burnycoder.github.io/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/"
KIND_LABEL = {"positive": "✅ positive", "negative": "❌ negative", "mixed": "🟡 mixed", "ground truth": "🔷 ground truth",
              "example": "⚪ example"}


def readme_gallery_md(figs: list[dict], png_dir: str = "reports/figures/viz", per_row: int = 2) -> str:
    """HTML table for the README: static PNGs (GitHub renders no JavaScript), each linking to its interactive
    version on the site, with the kind label and computed verdict under the title. The animated 3-D record is shown
    first, full width, as a GIF."""
    hero = "".join(f'<p align="center"><a href="{SITE_URL}#fig-{r["key"]}"><img src="{png_dir}/{r["key"]}.gif" '
                   f'alt="{r["title"]}" width="100%"></a><br><b>{r["title"]}</b> (animated here; interactive 3-D player '
                   f'on the site)<br>{KIND_LABEL[r["kind"]]}: {r["verdict"]}</p>\n\n' for r in figs if "pca3d" in r)
    cells = [f'<td width="50%" valign="top"><a href="{SITE_URL}#fig-{r["key"]}"><img src="{png_dir}/'
             f'{r.get("gallery_img", r["key"] + ".png")}" alt="{r["title"]}"></a><br><b>{r["title"]}</b><br>'
             f'{KIND_LABEL[r["kind"]]}: {r["verdict"]}</td>' for r in figs
             if "pca3d" not in r and r.get("in_gallery", True)]
    rows = ["<tr>" + "".join(cells[i:i + per_row]) + "</tr>" for i in range(0, len(cells), per_row)]
    return ("*Generated by `python -m phylogeo.run --phases viz_data,site` from `results/`; click a figure for the "
            f"interactive version on the [project site]({SITE_URL}) (hover, zoom, toggle series).*\n\n"
            + hero + "<table>\n" + "\n".join(rows) + "\n</table>\n")


def export_pngs(figs: list[dict], out: Path, changed: set[str] | None = None) -> dict[str, Path]:
    """Static PNG copy of every figure (kaleido, uses the installed Chrome) for the README and paper."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for r in figs:
        w, h = r["png_size"]
        p = out / f"{r['key']}.png"
        if changed is not None and r["key"] not in changed and p.is_file() and all(
                (out / f"{name}.gif").is_file() for name in r.get("gifs", {})):
            paths[r["key"]] = p
            continue
        r.get("png_fig", r["fig"]).write_image(p, width=w, height=h, scale=1.6)  # static layout may differ from the site's
        paths[r["key"]] = p
        for name, make in r.get("gifs", {}).items():  # animated records also get GIFs for the README
            paths[name + "_gif"] = make(out / f"{name}.gif")
    log.info("exported %d figure files (PNG, GIF) to %s", len(paths), out)
    return paths
