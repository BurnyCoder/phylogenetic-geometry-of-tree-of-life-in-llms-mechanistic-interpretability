"""Build the static GitHub Pages explorer (site/) from results/ (no hand-typed numbers).

Global context: the ``site`` phase copies the interactive Plotly figures and PNGs
produced by the analysis phases into ``site/assets/`` and writes ``site/index.html``
whose tables are rendered from the results JSON (via report.py's Markdown,
converted to HTML). Deployed by .github/workflows/pages.yml.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

from phylogeo.config import REPO_ROOT, RESULTS_DIR
from phylogeo.report import (conflict_table, eco_table, key_results_md, power_table, stage1a_table, stage1b_tables,
                             stage2_table, stage3_table, stage4_table, stage5_table, stage6_table)

SITE = REPO_ROOT / "site"
CSS = """
:root{color-scheme:light dark;--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e2e1dc;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:auto;padding:24px}
h1{font-size:1.7em}h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2em}
a{color:var(--accent)}small,.muted{color:var(--muted)}
table{border-collapse:collapse;font-size:13px;margin:12px 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}
img{max-width:100%;border:1px solid var(--line);border-radius:4px}
iframe{width:100%;height:820px;border:1px solid var(--line);border-radius:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
"""


def md_table_to_html(md: str) -> str:
    """Minimal Markdown -> HTML for the generated tables, bold text and paragraphs."""
    out, rows = [], []

    def flush():
        if rows:
            head, body = rows[0], [r for r in rows[2:]]
            out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>")
            rows.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            rows.append([html.escape(c.strip()) for c in line.strip("|").split("|")])
            continue
        flush()
        if line.strip():
            t = html.escape(line)
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
            t = re.sub(r"(?<![*\w])\*([^*]+?)\*(?![*\w])", r"<i>\1</i>", t)  # *italic*
            out.append(f"<p>{t}</p>")
    flush()
    return "\n".join(out)


def _copy(src: Path, dst_dir: Path) -> str | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    return f"assets/{dst_dir.name}/{src.name}"


def layer_profiles_html(run: str) -> str:
    """Interactive Plotly chart: TA_within vs relative layer depth for the main run, its controls
    and every stage-5 model (click legend entries to toggle). Data come from results JSON."""
    import json

    import plotly.graph_objects as go

    traces = {}
    s1b = RESULTS_DIR / run / "stage1b" / "results.json"
    if s1b.exists():
        for r, rec in json.loads(s1b.read_text())["models"].items():
            traces[f"{r} (stage 1b)"] = rec["ta_within_by_layer"]
    for f in sorted(RESULTS_DIR.glob("*/stage5/scale.json")):
        rec = json.loads(f.read_text())
        traces.setdefault(rec["run"], rec["ta_within_by_layer"])
    if not traces:
        return "<p>(not yet computed)</p>"
    fig = go.Figure()
    for name, y in traces.items():
        fig.add_scatter(x=[i / (len(y) - 1) for i in range(len(y))], y=y, mode="lines+markers", name=name,
                        marker={"size": 5}, line={"width": 2}, visible=True if run in name or "random" in name
                        or "step0" in name else "legendonly")
    fig.add_hline(y=1 / 3, line_dash="dot", annotation_text="chance 1/3")
    fig.update_layout(height=460, margin={"l": 50, "r": 10, "t": 30, "b": 40}, template="plotly_white",
                      xaxis_title="relative layer depth (0 = embeddings, 1 = last layer)", yaxis_title="TA_within",
                      legend={"font": {"size": 10}})
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def build_site(run: str = "qwen35_2b") -> Path:
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "assets").mkdir(parents=True)
    s1a = RESULTS_DIR / run / "stage1a"
    a_dir = SITE / "assets" / "stage1a"
    pca3d = _copy(s1a / "pca3d.html", a_dir)
    imgs = [_copy(p, a_dir) for p in sorted(s1a.glob("*.png"))]
    s1b_imgs = [_copy(p, SITE / "assets" / "stage1b") for p in sorted((RESULTS_DIR / run / "stage1b").glob("*.png"))]
    stage_imgs = {k: [_copy(p, SITE / "assets" / k) for p in sorted((RESULTS_DIR / run / k).glob("*.png"))]
                  for k in ("stage2", "stage4", "stage6", "eco")}
    s5_imgs = [_copy(p, SITE / "assets" / "stage5") for p in sorted((RESULTS_DIR / "stage5").glob("*.png"))]
    s2_imgs = stage_imgs["stage2"]

    def gallery(paths):
        return '<div class="grid">' + "".join(f'<a href="{p}"><img src="{p}" loading="lazy" alt="{Path(p).stem}"></a>'
                                             for p in paths if p) + "</div>"

    body = f"""
<h1>Phylogenetic geometry of the tree of life in an LLM</h1>
<p class="muted">Libor "Burny" Burian · Qwen3.5-2B · 400 animal species · Open Tree of Life reference ·
<a href="https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability">code &amp; paper</a> ·
<a href="https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/preregistration.md">preregistration</a></p>
<p>Do the hidden states an LLM computes for animal names reflect the tree of life? Each point below is one species
(mean residual-stream vector at the last name token over 8 prompts). Color = super-group (blue vertebrates, orange
arthropods, green other invertebrates, gray other), marker = class; thin gray lines = the reference tree drawn with
internal nodes at the centroid of their descendants. Use the slider to move through layers; hover for names.</p>
<h2>Headline results</h2>{md_table_to_html(key_results_md(run))}
<h2>Stage 1a — top-3 PCA, interactive</h2>
{f'<iframe src="{pca3d}" title="3D PCA of species representations"></iframe>' if pca3d else '<p>(not yet computed)</p>'}
<h2>Stage 1a — figures</h2>{gallery(imgs)}
<h2>Stage 1a — checks per layer</h2>{md_table_to_html(stage1a_table(run))}
<h2>Stage 1b — triplet accuracy (primary) and secondary hypotheses</h2>{md_table_to_html(stage1b_tables(run))}
{gallery(s1b_imgs)}
<h2>Explore: within-group triplet accuracy across layers and models</h2>
<p>Click legend entries to show or hide models (stage-5 models are hidden at first).</p>{layer_profiles_html(run)}
<h2>Phylogeny-vs-folk conflict triplets</h2>{md_table_to_html(conflict_table(run))}
<h2>Stage 2 — geometry and probes</h2>{md_table_to_html(stage2_table(run))}{gallery(s2_imgs)}
<h2>Stage 3 — behavior and causal tests</h2>{md_table_to_html(stage3_table(run))}
<h2>Stage 4 — ablations</h2>{md_table_to_html(stage4_table(run))}{gallery(stage_imgs["stage4"])}
<h2>Addendum A — habitat axis on held-out species</h2>{md_table_to_html(eco_table(run))}{gallery(stage_imgs["eco"])}
<h2>Stage 5 — other models: scale and training dynamics</h2>{md_table_to_html(stage5_table())}{gallery(s5_imgs)}
<h2>Stage 6 — sensitivity</h2>{md_table_to_html(stage6_table(run))}{gallery(stage_imgs["stage6"])}
<h2>Power simulation</h2>{md_table_to_html(power_table())}
<p class="muted">All numbers are generated from <code>results/</code> by <code>python -m phylogeo.run --phases report,site</code>.</p>
"""
    page = f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Phylogenetic geometry in LLMs</title><style>{CSS}</style></head><body>{body}</body></html>"
    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    return SITE / "index.html"
