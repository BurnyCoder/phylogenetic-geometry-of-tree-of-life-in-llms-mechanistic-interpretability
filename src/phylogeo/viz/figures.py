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
               "Top left is the shape a perfect tree code would give (the stage-1a synthetic positive control): nested "
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
    pt, ci, h1 = s["point_ta_within"], s["ci95_ta_within"], s["H1"]
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
    fig.add_scatter(x=[0.71], y=[DISPLAY["tree_oracle"]], mode="markers+text", name="reference oracle",
                    legendgroup="reference oracle", showlegend=False, text=[f"{pt['tree_oracle']:.2f} (off scale)"],
                    textposition="middle left", textfont={"color": AQUA, "size": 11},
                    marker={"color": AQUA, "symbol": "triangle-right", "size": 12},
                    hovertemplate=f"<b>{DISPLAY['tree_oracle']}</b><br>TA_within {pt['tree_oracle']:.3f} (off scale)"
                                  "<extra>reference oracle</extra>", row=1, col=1)
    fig.update_xaxes(range=[0.2, 0.72], title_text="TA_within (chance = 1/3)", row=1, col=1)
    req = h1["required_comparisons"]
    for r in req:
        lo, hi = h1["diff_ci95_vs"][r]
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
    sup = h1["supported"]
    return rec("h1", "Primary test (H1): tree order inside coarse groups", "positive" if sup else "negative",
               ("H1 supported" if sup else "H1 not supported") + f" (max-T p {_p(h1['maxT_p'])})",
               "Rooted-triplet accuracy on triplets inside one stratum whose names share no head noun, at the "
               f"cross-fitted layer {h1['modal_layer']}. The model beats every required untrained network and name "
               "statistic (panel B: every paired CI excludes zero), but by a modest margin; the exact tree would score 1.0 and the "
               "Linnaean-rank oracle shows what rank information alone gives.", fig, (1150, 640))


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
        f"; below chance in {', '.join(sorted(below))}" if below else "")
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
    for r, rec_ in s["models"].items():
        y = rec_["ta_within_by_layer"]
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
        y = r5["ta_within_by_layer"]
        fig.add_scatter(x=[i / (len(y) - 1) for i in range(len(y))], y=y, mode="lines", name=r5["run"],
                        legendgroup="stage 5", legendgrouptitle_text="stage-5 models (click to show)",
                        line={"color": LIGHTGRAY, "width": 1.5}, visible="legendonly",
                        hovertemplate=f"<b>{r5['run']}</b><br>TA_within %{{y:.3f}}<extra></extra>")
    _chance(fig)
    fig.update_xaxes(title_text="relative layer depth (0 = embeddings, 1 = last layer)")
    fig.update_yaxes(title_text="TA_within")
    _style(fig, 520).update_layout(legend={"font": {"size": 11}})
    y = s["models"][run]["ta_within_by_layer"]
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
    ph = s["exploratory"]["posthoc_vs_untrained"]
    rows = [("S2_within_order", "S2: inside taxonomic orders"), ("S3_lexical_discordant", "S3: spelling points the wrong way"),
            ("S4_cooccurrence_discordant", "S4: GloVe co-occurrence points the wrong way"),
            ("char3_tie_lexically_neutral", "spelling neutral (post hoc)")]
    fig = go.Figure()
    for i, (k, label) in enumerate(rows):
        v = ph[k]
        lo, hi = v["qwen35_2b_ci95"]
        beats = all(v[f"diff_ci95_vs_{c}"][0] > 0 for c in ["qwen35_2b_random", "pythia410m_step0"])
        ylab = f"{label}<br>{v['n_triplets']} triplets · " + ("beats both untrained ✓" if beats else "≈ untrained ✗")
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
    return rec("secondary", "Beyond spelling and co-occurrence? Model vs untrained networks", "negative",
               "S2–S4 matched by untrained networks",
               "Registered secondary subsets (S2–S4) and a post-hoc spelling-neutral subset. Where spelling or GloVe "
               "co-occurrence disagree with the tree, the trained model does no better than untrained networks, so "
               "phylogenetic information beyond those statistics is not shown. It does beat them where spelling has "
               "no preference.", fig, (1100, 520))


def fig_conflict(run) -> dict | None:
    c = _load(RESULTS_DIR / run / "conflict" / "conflict.json")
    if not c:
        return None
    reps = [r for r in ["qwen35_2b", "qwen35_2b_random", "pythia410m_step0", "pythia410m", "char3", "glove"]
            if r in c["summary"]]
    trip = [t for t in c["triplets"] if t.get("resolved")]
    ylab = [f"{t['anchor']}: <b>{t['tree_closer']}</b> or {t['other']}?" for t in trip]
    Z = np.array([[t[r][c["summary"][r]["layer"]] if len(t[r]) > 1 else t[r][0] for r in reps] for t in trip])
    xlab = [f"{DISPLAY[r]}<br>{c['summary'][r]['phylo_choice']}/{c['summary'][r]['n']} tree" for r in reps]
    fig = go.Figure(go.Heatmap(z=Z, x=xlab, y=ylab, colorscale=[[0, ORANGE], [0.5, ORANGE], [0.5, BLUE], [1, BLUE]],
                               zmin=0, zmax=1, xgap=2, ygap=2, showscale=False,
                               text=np.where(Z == 1, "tree", "other"), texttemplate="%{text}",
                               textfont={"size": 10, "color": "white"},
                               customdata=np.array([[t["type"]] * len(reps) for t in trip]),
                               hovertemplate="%{y}<br>%{x}<br>chose the <b>%{text}</b> partner<br>type: %{customdata}"
                                             "<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont={"size": 11})
    fig.update_xaxes(side="top", tickfont={"size": 11})
    _style(fig, 60 + 24 * len(trip), legend=False).update_layout(margin={"l": 330, "t": 90, "b": 20})
    s = c["summary"][run]
    return rec("conflict", "Phylogeny-vs-folk conflict triplets (ground truth in bold)", "negative",
               f"tree partner in {s['phylo_choice']}/{s['n']}, like untrained networks",
               "For each hand-picked triplet the bold name is the tree-closer partner (ground truth from the Open Tree "
               "topology). Blue = the representation is closer to the tree partner, orange = closer to the folk / "
               "ecological / lexical partner. Neural representations at the confirmatory layer; character 3-grams and "
               "GloVe as name statistics. The trained model mostly follows the folk partner, and so do untrained "
               "networks.", fig, (1100, 80 + 24 * len(trip)))


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
    fig.add_hline(pr["raw_angular_within_stratum_spearman_all_species"], line_dash="dash", line_color=BLUE, row=1,
                  col=2, annotation_text="raw distances, no probe", annotation_font_color=BLUE, annotation_font_size=11)
    fig.update_yaxes(title_text="held-out families (5-fold)", row=1, col=1)
    _style(fig, 520, legend=False)
    beaten = pr["class_blob"]["test_test_pearson"][0] >= pr["model"]["test_test_pearson"][0]
    return rec("probe", "Goodfire-style 10-d flat probe: tree or coarse groups?", "negative" if beaten else "positive",
               "class-only input does as well" if beaten else "beats the class-only input",
               "Probe trained to predict tree distance from representations, evaluated on held-out families (error bars "
               "= SD over seeds). An input that carries only class membership (aqua) matches or beats the model on all "
               "pairs, so that headline correlation mostly measures coarse groups. Inside strata the probe adds little "
               "over raw distances.", fig, (1100, 540))


def fig_treeness(run) -> dict | None:
    s2 = _load(RESULTS_DIR / run / "stage2" / "results.json")
    if not s2:
        return None
    d, nj = s2["delta_rel"], s2["nj_rf"]
    fig = make_subplots(1, 2, horizontal_spacing=0.14,
                        subplot_titles=["A. Gromov δ_rel (0 = perfect tree metric)",
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
    a, w, cc, hp = s3["C_abl"], s3["C_swap"], s3["class_control"], s3["causal_holm"]
    fig = make_subplots(1, 2, horizontal_spacing=0.12,
                        subplot_titles=[f"A. Ablation (C-abl): Holm p {_p(hp['C_abl'])}",
                                        f"B. Interchange (C-swap): Holm p {_p(hp['C_swap'])}"])
    fig.add_histogram(x=a["random_mean_drops"], nbinsx=25, marker={"color": GRAY, "line": {"color": SURFACE, "width": 1}},
                      name=f"{a['n_random_subspaces']} random subspaces", legendgroup="rand",
                      hovertemplate="drop %{x}<br>%{y} subspaces<extra>random</extra>", row=1, col=1)
    fig.add_histogram(x=w["switch_rate_random"], marker={"color": GRAY, "line": {"color": SURFACE, "width": 1}},
                      showlegend=False, legendgroup="rand", xbins={"size": 1 / w["n_pairs"]},
                      hovertemplate="switch rate %{x}<br>%{y} subspaces<extra>random</extra>", row=1, col=2)
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
    fig.update_xaxes(title_text="share of 200 pairs whose class answer switches", row=1, col=2)
    fig.update_yaxes(title_text="random subspaces", row=1, col=1)
    _legend_below(_style(fig, 520)).update_layout(barmode="overlay")
    spec = cc["phylogeny_specific"]
    return rec("causal", "Causal tests: does the model use the probe subspace?", "positive" if spec else "mixed",
               ("C-swap " + ("✓" if hp["C_swap"] < 0.05 else "✗") + ", C-abl " + ("✓" if hp["C_abl"] < 0.05 else "✗")
                + ", " + ("phylogeny-specific" if spec else "not phylogeny-specific")),
               "Gray: the null distribution from variance-matched random 10-d subspaces at the same layer, on held-out "
               "species. Blue: the phylogeny probe subspace. Orange: a probe trained on class membership only. Ablation "
               "of the probe is inside the random tail; interchange beats every random subspace but switches few "
               "answers, and the class-only subspace does as well on both: the subspace is used for coarse class "
               "answers, not shown to be used for phylogeny.", fig, (1100, 520))


def fig_habitat(run) -> dict | None:
    e = _load(RESULTS_DIR / run / "eco" / "results.json")
    if not e:
        return None
    hs = e["heldout_scores"]
    mam = [h for h in hs if h["stratum"] == "mammals" and h.get("forstrat_mammal")]
    bird = [h for h in hs if h["stratum"] == "birds" and h.get("water_pct_bird") not in ("", None)]
    t = e["tests"]
    lab = {"M": "marine", "G": "ground", "Ar": "arboreal", "A": "aerial", "S": "scansorial"}
    fig = make_subplots(1, 2, horizontal_spacing=0.12,
                        subplot_titles=[f"A. Held-out mammals by foraging stratum (marine < ground: Holm p "
                                        f"{_p(t['M1_marine_lt_ground']['holm_p'])})",
                                        f"B. Held-out birds: water foraging (ρ = {t['B1_water_negative']['rho']:.2f}, "
                                        f"Holm p {_p(t['B1_water_negative']['holm_p'])})"])
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
    fig.update_xaxes(title_text="% of diet foraged in or on water (EltonTraits)", row=1, col=2)
    _style(fig, 520, legend=False)
    aq = t["M1_marine_lt_ground"]["holm_p"] < 0.05 and t["B1_water_negative"]["holm_p"] < 0.05
    ae = t["B2_aerial_positive"]["holm_p"] < 0.05 and t["M2_aerial_gt_ground"].get("holm_p", 1) < 0.05
    return rec("habitat", "Addendum A: an aquatic-to-terrestrial axis, tested on held-out species",
               "positive" if aq and ae else "mixed" if aq or ae else "negative",
               f"aquatic end {'confirmed' if aq else 'not confirmed'}, aerial end {'confirmed' if ae else 'not confirmed'}",
               "The axis was found post hoc in stage 1a and then tested as a registered directional hypothesis on "
               "species not used to find it. Marine mammals (orange) and water-foraging birds score low; the aerial "
               "prediction failed (no bats among held-out mammals; aerial birds do not score high). Names with water "
               "words were not controlled.", fig, (1100, 540))


def fig_scale(run) -> dict | None:
    s5 = _load(RESULTS_DIR / "stage5" / "summary.json")
    if not s5:
        return None
    runs = s5["runs"]
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
    for name, v in s5["reference_ta_within"].items():
        fig.add_hline(v, line_dash="dot", line_color=LIGHTGRAY, row=1, col=1, annotation_text=DISPLAY.get(name, name),
                      annotation_font_size=10, annotation_font_color=MUTED, annotation_position="bottom right")
    for col in (1, 2):
        _chance(fig, "y", 1, col)
    fig.update_xaxes(type="log", title_text="parameters (log)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="training step (log; step 0 plotted at 0.5)", row=1, col=2)
    fig.update_yaxes(title_text="cross-fitted TA_within", row=1, col=1)
    _legend_below(_style(fig, 540))
    return rec("scale", "More models: grows with scale, learned during training", "positive", "descriptive (stage 5)",
               "Same species, triplets and metric for every model (199 permutations, lighter than stage 1b). Dotted "
               "lines: untrained and name-statistic references. Within each family the effect grows with size; in "
               "Pythia-1.4B it stays near chance up to step 512 and appears by step 1000.", fig, (1150, 560))


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
    return rec("sensitivity", "Sensitivity: seeds, templates, species subsamples", "positive", "robust",
               f"{n} repetitions per perturbation. On every draw where controls were recomputed, the model stays above "
               "the strongest untrained or name-statistic control.", fig, (1100, 500))


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
               "tree distance broken by table order).", fig, (1100, 400))


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


def _cloud_traces(coords: np.ndarray, names, strata, tree, scene: str, showlegend: bool, lim: float) -> list:
    """Scatter3d traces for one panel: the reference tree drawn with internal nodes at the centroid of their
    descendant leaves (as in stage 1a, computed from all species), then one marker trace per stratum
    (colour = super-group). Points outside the shared box are not drawn. Legend groups are shared across
    panels, so one legend click toggles a stratum in both."""
    from phylogeo.viz.pca import STRATUM_STYLE, overlay_positions

    pos = _clip(overlay_positions(tree, coords), lim)
    coords = _clip(coords, lim)
    seg = [v for v in range(tree.N) if v != tree.root]
    xs, ys, zs = ([c for v in seg for c in (pos[tree.parent[v], k], pos[v, k], None)] for k in range(3))
    out = [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line={"color": "#b9b8b2", "width": 1.5}, opacity=0.5,
                        name="reference tree (centroids)", legendgroup="tree", showlegend=showlegend, hoverinfo="skip",
                        scene=scene)]
    strata, names = np.asarray(strata), np.asarray(names)
    for s, (color, _, sym) in STRATUM_STYLE.items():
        m = strata == s
        if m.any():
            out.append(go.Scatter3d(x=coords[m, 0], y=coords[m, 1], z=coords[m, 2], mode="markers", name=s,
                                    legendgroup=s, showlegend=showlegend, text=names[m], scene=scene,
                                    marker={"color": color, "symbol": sym, "size": 3.5,
                                            "line": {"color": SURFACE, "width": 0.5}},
                                    hovertemplate="<b>%{text}</b><br>" + s + "<extra></extra>"))
    return out


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
    r = rec("pca3d", "Top-3 principal components in 3-D, animated through all layers: trained vs random init", "mixed",
            f"trained: H1a checks tree-consistent in {k}/{len(checks or [])} layers; by eye: class arms, no nested "
            "branching",
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
    (``steps`` interpolated frames per layer) while both cameras turn together; rendered with kaleido in one batch
    and assembled with Pillow (https://pillow.readthedocs.io/)."""
    import tempfile

    import plotly.io as pio
    from PIL import Image

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
    with tempfile.TemporaryDirectory() as tmp:
        files = [Path(tmp) / f"f{j:03d}.png" for j in range(len(figs))]
        pio.write_images(figs, files, width=size[0], height=size[1], scale=1)
        frames = [Image.open(f).convert("RGB").quantize(colors=80, method=Image.Quantize.MEDIANCUT) for f in files]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True)
    log.info("animated GIF (%d frames) written to %s", len(figs), path)
    return path


def build_figures(run: str = "qwen35_2b") -> list[dict]:
    """All figure records whose inputs exist, in reading order (ground truth first)."""
    from phylogeo.phases import load_species

    df, tree = load_species()
    builders = [lambda: fig_pca3d(run, tree), lambda: fig_taxonomy(df), lambda: fig_distance_heatmaps(run, df),
                lambda: fig_expected_pca(run),
                lambda: fig_h1_forest(run), lambda: fig_layer_profile(run), lambda: fig_per_stratum(run),
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
    cells = [f'<td width="50%" valign="top"><a href="{SITE_URL}#fig-{r["key"]}"><img src="{png_dir}/{r["key"]}.png" '
             f'alt="{r["title"]}"></a><br><b>{r["title"]}</b><br>{KIND_LABEL[r["kind"]]}: {r["verdict"]}</td>'
             for r in figs if "pca3d" not in r]
    rows = ["<tr>" + "".join(cells[i:i + per_row]) + "</tr>" for i in range(0, len(cells), per_row)]
    return ("*Generated by `python -m phylogeo.run --phases viz_data,site` from `results/`; click a figure for the "
            f"interactive version on the [project site]({SITE_URL}) (hover, zoom, toggle series).*\n\n"
            + hero + "<table>\n" + "\n".join(rows) + "\n</table>\n")


def export_pngs(figs: list[dict], out: Path) -> dict[str, Path]:
    """Static PNG copy of every figure (kaleido, uses the installed Chrome) for the README and paper."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for r in figs:
        w, h = r["png_size"]
        p = out / f"{r['key']}.png"
        r["fig"].write_image(p, width=w, height=h, scale=1.6)
        paths[r["key"]] = p
        if "pca3d" in r:  # the animated record also gets a GIF for the README
            from phylogeo.phases import load_species

            paths[r["key"] + "_gif"] = export_pca3d_gif(r, load_species()[1], out / f"{r['key']}.gif")
    log.info("exported %d figure files (PNG, GIF) to %s", len(paths), out)
    return paths
