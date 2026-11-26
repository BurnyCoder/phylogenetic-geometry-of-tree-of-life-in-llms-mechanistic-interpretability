"""Build the static GitHub Pages explorer (site/) from results/ (no hand-typed numbers).

Global context: the ``site`` phase
1. builds the labelled interactive figures (``viz/figures.py``), embeds them as Plotly HTML cards with a
   verdict badge, and exports a PNG of each to ``reports/figures/viz/``;
2. injects a gallery of those PNGs, each linking to its interactive card, near the top of README.md, and the
   animated 3-D within-group view next to the H1 highlight (``WITHIN3D`` block);
3. adds a nearest-neighbour explorer for every species (data: ``results/<run>/viz/neighbors.json``);
4. keeps the generated tables and the stage PNGs below (report.py's Markdown converted to HTML).
Deployed by .github/workflows/pages.yml.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

from phylogeo.config import REPO_ROOT, RESULTS_DIR
from phylogeo.report import (conflict_table, eco_table, inject_block, key_results_md, power_table, stage1a_table,
                             stage1b_tables, stage2_table, stage3_table, stage4_table, stage5_table, stage6_table)

SITE = REPO_ROOT / "site"
PNG_DIR = REPO_ROOT / "reports" / "figures" / "viz"
CSS = """
:root{color-scheme:light dark;--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e2e1dc;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1200px;margin:auto;padding:24px}
h1{font-size:1.7em}h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2em}
a{color:var(--accent)}small,.muted{color:var(--muted)}
table{border-collapse:collapse;font-size:13px;margin:12px 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}
img{max-width:100%;border:1px solid var(--line);border-radius:4px}
iframe{width:100%;height:820px;border:1px solid var(--line);border-radius:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
/* figure cards keep the light chart surface in both colour schemes (charts are drawn for it) */
.card{background:#fcfcfb;color:#0b0b0b;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:20px 0}
.card h3{margin:2px 0 6px;font-size:1.15em}.card .cap{color:#52514e;font-size:13.5px;margin:6px 0 2px}
.card a{color:#2a78d6}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;color:#fff;margin-right:6px}
.k-positive{background:#0ca30c}.k-negative{background:#d03b3b}.k-mixed{background:#fab219;color:#0b0b0b}
.k-ground-truth{background:#2a78d6}.k-example{background:#8a8984}
.nav{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}.nav a{text-decoration:none;font-size:12.5px;padding:3px 9px;
border:1px solid var(--line);border-radius:6px;color:var(--fg)}
.nn{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:10px}
.nn h4{margin:0 0 4px;font-size:13.5px}.chip{display:block;padding:3px 8px;margin:2px 0;border-radius:5px;font-size:12.5px}
.key span{display:inline-block;padding:2px 8px;border-radius:5px;font-size:12px;margin:2px}
input{font:inherit;padding:4px 8px;border:1px solid #c3c2b7;border-radius:6px;min-width:260px}
"""
ICON = {"positive": "✓", "negative": "✗", "mixed": "◐", "ground truth": "◆", "example": "●"}


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
            t = re.sub(r"\[([^\]]+)\]\((https://[^\s)]+)\)", r'<a href="\2">\1</a>', t)
            out.append(f"<p>{t}</p>")
    flush()
    return "\n".join(out)


def _copy(src: Path, dst_dir: Path) -> str | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    return f"assets/{dst_dir.name}/{src.name}"


def badge(kind: str, verdict: str) -> str:
    """Status-coloured pill with an icon and text, so the verdict never relies on colour alone."""
    return (f'<span class="badge k-{kind.replace(" ", "-")}">{ICON[kind]} {html.escape(kind)}</span>'
            f"<b>{html.escape(verdict)}</b>")


def figure_cards(figs: list[dict]) -> tuple[str, str]:
    """(jump menu, cards) for the figure records. Plotly.js is loaded once in the page head (see build_site);
    a record with ``html`` (the animated 3-D player) supplies its own markup instead of a Plotly figure."""
    nav = '<div class="nav">' + "".join(f'<a href="#fig-{r["key"]}">{ICON[r["kind"]]} {html.escape(r["title"])}</a>'
                                        for r in figs) + '<a href="#fig-explorer">● Explore neighbours</a></div>'
    cards = []
    cfg = {"responsive": True, "displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}
    for r in figs:
        div = r.get("html") or r["fig"].to_html(include_plotlyjs=False, full_html=False, config=cfg,
                                                div_id=f"plot-{r['key']}")
        cards.append(f'<div class="card" id="fig-{r["key"]}"><h3>{html.escape(r["title"])}</h3>'
                     f'<div>{badge(r["kind"], r["verdict"])}</div>{div}<p class="cap">{html.escape(r["caption"])} '
                     f'<a href="https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-'
                     f'interpretability/blob/main/reports/figures/viz/{r["key"]}.png">PNG</a></p></div>')
    return nav, "\n".join(cards)


def explorer_html(run: str) -> str:
    """Nearest-neighbour explorer: type any species, see its nearest species under the tree (ground truth),
    the model and the untrained controls, coloured by the closest shared rank."""
    from phylogeo.viz.figures import DISPLAY, LEVEL_COLORS

    f = RESULTS_DIR / run / "viz" / "neighbors.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    reps = ["tree", run, *[r for r in d["neighbors"] if r not in ("tree", run)]]
    data = {"names": d["names"], "strata": d["strata"], "tax": d["taxonomy"], "levels": d["levels"],
            "nb": d["neighbors"], "reps": reps, "labels": [DISPLAY.get(r, r) for r in reps],
            "levelNames": d["level_names"], "colors": LEVEL_COLORS}
    key = "".join(f'<span style="background:{c};color:{"#fff" if i < 3 else "#0b0b0b"}">{n}</span>'
                  for i, (c, n) in enumerate(zip(LEVEL_COLORS, d["level_names"])))
    options = "".join(f'<option value="{html.escape(n)}">' for n in sorted(d["names"], key=str.lower))
    return f"""<div class="card" id="fig-explorer"><h3>Explore: the nearest species of any species</h3>
<div>{badge("example", f"all {len(d['names'])} species, layer {d['layer']}")}</div>
<p class="cap">Type or pick a species. Each column lists its {d['k']} nearest species under one representation; the
colour is the closest Linnaean rank it shares with the query. The first column is the ground truth (reference tree;
ties in tree distance keep table order).</p>
<input list="species-list" id="nn-q" value="lion" aria-label="species"><datalist id="species-list">{options}</datalist>
<span id="nn-tax" class="cap"></span><div class="key">{key}</div><div class="nn" id="nn-out"></div></div>
<script>
const NN = {json.dumps(data)};
function nnRender() {{
  const q = document.getElementById('nn-q').value, i = NN.names.indexOf(q);
  if (i < 0) return;
  const t = NN.tax;
  document.getElementById('nn-tax').textContent = ' ' + t.sci_name[i] + ' · ' + [t.genus[i], t.family[i], t.order[i],
    t['class'][i], t.phylum[i]].filter(Boolean).join(' · ');
  document.getElementById('nn-out').innerHTML = NN.reps.map((r, k) => {{
    const lv = NN.levels[r][i], nb = NN.nb[r][i], close = lv.filter(v => v <= 2).length;
    return '<div><h4>' + NN.labels[k] + '<br><small>' + close + '/' + lv.length + ' same order or closer</small></h4>' +
      nb.map((j, m) => '<span class="chip" title="' + NN.levelNames[lv[m]] + ' · ' + NN.strata[j] + '" style="background:' +
        NN.colors[lv[m]] + ';color:' + (lv[m] < 3 ? '#fff' : '#0b0b0b') + '">' + (m + 1) + '. ' + NN.names[j] +
        '</span>').join('') + '</div>';
  }}).join('');
}}
document.getElementById('nn-q').addEventListener('input', nnRender); nnRender();
</script>"""


def build_site(run: str = "qwen35_2b", *, reuse_media: bool = False) -> Path:
    from phylogeo.viz.figures import build_figures, export_pngs, group_view_md, readme_gallery_md, within3d_md

    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    figs = build_figures(run)
    changed = {"h1", "secondary", "conflict", "causal", "habitat", "probe", "scale", "sensitivity", "layers", "treeness"} if reuse_media else None
    # Kaleido's long-lived renderer writes all charts before closing its one browser.
    # https://github.com/plotly/Kaleido
    import kaleido
    kaleido.start_sync_server(n=1, timeout=120)
    try:
        export_pngs(figs, PNG_DIR, changed)
    finally:
        kaleido.stop_sync_server()
    inject_block(REPO_ROOT / "README.md", readme_gallery_md(figs), tag="FIGURES")
    for r in figs:  # the within-group views also sit at the top of the README's visual summary
        if r["key"] == "within3d":
            inject_block(REPO_ROOT / "README.md", within3d_md(r), tag="WITHIN3D")
        elif r["key"] == "view_birds":
            inject_block(REPO_ROOT / "README.md", group_view_md(r), tag="BIRDS3D")
    nav, cards = figure_cards(figs)
    s1a = RESULTS_DIR / run / "stage1a"
    a_dir = SITE / "assets" / "stage1a"
    pca3d = _copy(s1a / "pca3d.html", a_dir)
    imgs = [_copy(p, a_dir) for p in sorted(s1a.glob("*.png"))]
    s1b_imgs = [_copy(p, SITE / "assets" / "stage1b") for p in sorted((RESULTS_DIR / run / "stage1b").glob("*.png"))]
    stage_imgs = {k: [_copy(p, SITE / "assets" / k) for p in sorted((RESULTS_DIR / run / k).glob("*.png"))]
                  for k in ("stage2", "stage4", "stage6", "eco")}
    s5_imgs = [_copy(p, SITE / "assets" / "stage5") for p in sorted((RESULTS_DIR / "stage5").glob("*.png"))]

    def gallery(paths):
        return '<div class="grid">' + "".join(f'<a href="{p}"><img src="{p}" loading="lazy" alt="{Path(p).stem}"></a>'
                                             for p in paths if p) + "</div>"

    examples = ""
    if run == "qwen35_2b" and (RESULTS_DIR / run / "examples/neighbors.json").is_file():
        from phylogeo.verification.publication import neighbor_details
        examples = '<h2 id="nearest-neighbor-examples">Nearest-neighbor examples</h2>' + md_table_to_html(neighbor_details())

    kinds = "".join(f'{badge(k, "")} ' for k in ICON)
    body = f"""
<h1>Phylogenetic geometry of the tree of life in an LLM</h1>
<p class="muted">Libor "Burny" Burian · Qwen3.5-2B · 400 animal species · Open Tree of Life reference ·
<a href="https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability">code &amp; paper</a> ·
<a href="https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/preregistration.md">preregistration</a></p>
<p>Do the hidden states an LLM computes for animal names reflect the tree of life? Each species is represented by the
residual-stream vector at the last token of its name, averaged over 8 prompts. The figures below compare that geometry
with the reference tree (ground truth), with untrained networks and with simple name statistics. Every chart is
interactive (hover, zoom, click legend entries) and labelled with its verdict, computed from the results files.</p>
<h2>Independent reanalysis of archived observations</h2>
<p>Triplet accuracy measures agreement with the sampled reference topology. The primary family-block criterion was met.
The order-block interval against Pythia step 0 includes zero. A paired causal switch comparison remains unevaluable.
<a href="https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-scientific-audit.md">Audit and sources</a>.</p>
<h2>Headline results</h2>{md_table_to_html(key_results_md(run))}
<h2>Visual summary</h2><p>{kinds}</p>{nav}
{cards}
{explorer_html(run)}
<p>Stage-specific legacy image galleries and exploratory tables below preserve the original run conventions. The corrected tables above and the audit give the current hypothesis decisions.</p>
<h2>Stage 1a — top-3 PCA, interactive 3D</h2>
<p>Each point is one species; colour = super-group (blue vertebrates, orange arthropods, aqua other invertebrates,
gray other), marker = stratum; thin gray lines = the reference tree with internal nodes at the centroid of their
descendants. Use the slider to move through layers.</p>
{f'<iframe src="{pca3d}" title="3D PCA of species representations"></iframe>' if pca3d else '<p>(not yet computed)</p>'}
<h2>Stage 1a — figures</h2>{gallery(imgs)}
<h2>Stage 1a — checks per layer</h2>{md_table_to_html(stage1a_table(run))}
<h2>Stage 1b — triplet accuracy (primary) and secondary hypotheses</h2>{md_table_to_html(stage1b_tables(run))}
{gallery(s1b_imgs)}
{examples}
<h2>Phylogeny-vs-folk conflict triplets</h2>{md_table_to_html(conflict_table(run))}
<h2>Stage 2 — geometry and probes</h2>{md_table_to_html(stage2_table(run))}{gallery(stage_imgs["stage2"])}
<h2>Stage 3 — behavior and causal tests</h2>{md_table_to_html(stage3_table(run))}
<h2>Stage 4 — ablations</h2>{md_table_to_html(stage4_table(run))}{gallery(stage_imgs["stage4"])}
<h2>Addendum A — habitat axis on held-out species</h2>{md_table_to_html(eco_table(run))}{gallery(stage_imgs["eco"])}
<h2>Stage 5 — other models: scale and training dynamics</h2>{md_table_to_html(stage5_table())}{gallery(s5_imgs)}
<h2>Stage 6 — sensitivity</h2>{md_table_to_html(stage6_table(run))}{gallery(stage_imgs["stage6"])}
<h2>Power simulation</h2>{md_table_to_html(power_table())}
<p class="muted">All numbers are generated from <code>results/</code> by
<code>uv run --frozen python -m phylogeo.verification --render-only</code>.</p>
"""
    from plotly.offline import get_plotlyjs_version

    plotly_js = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"  # same build plotly.py targets
    page = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Phylogenetic geometry in LLMs</title>"
            f"<script src='{plotly_js}' charset='utf-8'></script><style>{CSS}</style></head><body>{body}</body></html>")
    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    return SITE / "index.html"
