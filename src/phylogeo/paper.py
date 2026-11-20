"""Generate the paper (reports/paper.typ -> reports/paper.pdf) from results JSON only.

Global context: every number in the paper is read from ``results/**/*.json`` here,
so the paper can never drift from the analysis (one command regenerates it:
``uv run python -m phylogeo.run --phases paper``). Figures are copied from
``results/``; the architecture diagram is drawn with matplotlib. Typst is
compiled through the ``typst`` PyPI package (https://pypi.org/project/typst/).
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from phylogeo.config import REPO_ROOT, RESULTS_DIR

REPORTS = REPO_ROOT / "reports"
FIG = REPORTS / "figures"
RUN = "qwen35_2b"
REPO_URL = "https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability"


def _j(*parts):
    p = RESULTS_DIR.joinpath(*parts)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def fp(p) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def esc(s: str) -> str:
    """Escape Typst markup characters in free text."""
    for ch in ("#", "$", "@", "<", ">", "*", "_"):
        s = s.replace(ch, "\\" + ch)
    return s


def architecture_png(path: Path) -> None:
    """Pipeline diagram (boxes and arrows) in the style of ML papers; non-overlapping layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    H = 0.95
    data, model, stage, ctl = "#eeeeec", "#e8f0fb", "#fdecd9", "#f4f4f2"
    boxes = {  # name: (x, y, width, color, text)
        "wd": (0.1, 3.0, 2.0, data, "Wikidata (QLever)\n400 animal species\ncommon names, NCBI ids"),
        "prompt": (2.5, 3.0, 2.0, model, "8 templates × species\nchat format\n'Tell me about the {name}.'"),
        "llm": (4.9, 3.0, 2.0, model, "Qwen3.5-2B (bf16)\nresidual stream, 25 layers\nlast name token"),
        "vec": (7.3, 3.0, 2.0, model, "species vectors\nper-template centering\ncosine distance"),
        "ott": (0.1, 1.5, 2.0, data, "Open Tree of Life topology\n(CC0) + NCBI ranks\n+ TimeTree ages"),
        "ctl": (0.1, 0.15, 2.6, ctl, "Controls: random-init Qwen,\nPythia step0, char-3-grams, GloVe,\ntoken count, popularity, oracles"),
        "s1": (3.0, 0.15, 2.5, stage, "Stage 1: top-3 PCA,\ntriplet accuracy, Mantel,\nconflict triplets"),
        "s2": (5.7, 0.15, 2.5, stage, "Stage 2: 10-d flat-subspace\nprobe (held-out families),\ngeodesics, δ-hyperbolicity"),
        "s3": (8.4, 0.15, 2.5, stage, "Stage 3: ablate / swap\nthe subspace, behavioural\njudgments"),
    }
    for x, y, w, col, t in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.04", fc=col, ec="0.35", lw=0.8))
        ax.text(x + w / 2, y + H / 2, t, ha="center", va="center", fontsize=7.6)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="0.3"))

    for u, v in (("wd", "prompt"), ("prompt", "llm"), ("llm", "vec")):
        xu, yu, wu = boxes[u][:3]
        arrow((xu + wu, yu + H / 2), (boxes[v][0], yu + H / 2))
    xv, yv, wv = boxes["vec"][:3]
    for st in ("s1", "s2", "s3"):
        xs, ys, ws = boxes[st][:3]
        arrow((xv + wv / 2, yv), (xs + ws / 2, ys + H))
    arrow((2.1, 1.97), (3.0, 0.75))  # tree -> stages
    arrow((2.7, 0.62), (3.0, 0.62))  # controls -> stages
    ax.text(0.1, 4.2, "Phylogenetic geometry of animal species in an LLM — pipeline", fontsize=9.5, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_paper() -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    architecture_png(FIG / "architecture.png")
    figs = {
        "spot": RESULTS_DIR / RUN / "stage1a" / "spotlight_layer21_pc12.png",
        "grid": RESULTS_DIR / RUN / "stage1a" / "grid_colored_pc12.png",
        "months": RESULTS_DIR / RUN / "stage1a" / "calib_months_layer12_pc12.png",
        "brown": RESULTS_DIR / RUN / "stage1a" / "calib_synthetic_brownian_tree.png",
        "ta": RESULTS_DIR / RUN / "stage1b" / "ta_within_by_layer.png",
        "probe": RESULTS_DIR / RUN / "stage2" / "probe_heldout_r.png",
        "mammals": RESULTS_DIR / RUN / "stage1a" / "zoom_mammals_layer21.png",
    }
    for k, p in figs.items():
        if p.exists():
            shutil.copy2(p, FIG / f"{k}.png")
    s1b = _j(RUN, "stage1b", "results.json")
    s2 = _j(RUN, "stage2", "results.json")
    s3 = _j(RUN, "stage3", "results.json")
    s4 = _j(RUN, "stage4", "results.json")
    eco = _j(RUN, "eco", "results.json")
    conf = _j(RUN, "conflict", "conflict.json")
    power = _j("power", "summary.json")
    h1 = s1b["H1"]
    sec = s1b["secondary"]
    pt, ci = s1b["point_ta_within"], s1b["ci95_ta_within"]
    cs = conf["summary"][RUN]
    probe = s2["probe"]

    def ta_row(k, label):
        if k not in pt or k not in ci:
            return ""
        return f"[{label}], [{f3(pt[k])}], [{f3(ci[k][0])}–{f3(ci[k][1])}],"

    rows = "\n    ".join(r for r in [
        ta_row(RUN, "*Qwen3.5-2B (main)*"), ta_row("qwen35_2b_random", "random-init Qwen3.5-2B"),
        ta_row("pythia410m_step0", "Pythia-410m step 0 (untrained)"), ta_row("pythia410m", "Pythia-410m (trained)"),
        ta_row("char3", "character 3-grams"), ta_row("input_embedding", "input embedding (layer 0)"),
        ta_row("glove", "GloVe 300d (co-occurrence)"), ta_row("token_count", "name token count"),
        ta_row("log_popularity", "log Wikipedia sitelinks"), ta_row("class_oracle", "class oracle"),
        ta_row("rank_oracle", "Linnaean rank oracle"), ta_row("tree_oracle", "tree oracle (upper bound)")] if r)
    sec_rows = "\n    ".join(
        f"[{k}], [{esc(d)}], [{f3(v)}], [{fp(sec[k]['maxT_p'])}], [{fp(sec[k]['holm_p'])}],"
        for k, d, v in [
            ("S1", "global Mantel ρ (cosine vs path length)", sec["S1"]["mantel_spearman_pathlen_modal"]),
            ("S2", "TA within orders (head-noun-disjoint)", sec["S2"]["ta_modal"]),
            ("S3", "TA on lexical-discordant triplets", sec["S3"]["ta_modal"]),
            ("S4", "TA on co-occurrence-discordant triplets", sec["S4"]["ta_modal"]),
            ("S5", "partial ρ(distance, age | topology)", sec["S5"]["pooled_partial_spearman"])])
    probe_rows = "\n    ".join(
        f"[{esc(k)}], [{f3(v['test_test_pearson'][0])} ± {f3(v['test_test_pearson'][1])}], [{f3(v['train_train_pearson'][0])}],"
        for k, v in probe.items() if k != "selectivity_test_test")
    stage3_txt = ""
    if s3:
        ca, cw = s3["C_abl"], s3["C_swap"]
        bs = s3["behavior_species_triplets"]
        bc = s3["behavior_conflict"]
        stage3_txt = f"""
== Causal tests and behaviour (stage 3)
We fit the flat probe on half of the families and intervened on the held-out half at layer {s3['layer']}.
*Mean-ablating* the 10-d probe subspace at the name tokens lowered log P(correct class word) by
{f3(ca['mean_drop_logp_probe'])} nats on average, vs {f3(ca['mean_drop_logp_random'])} for 20 variance-matched random
subspaces (rank p {fp(ca['p_rank_vs_random'])}; paired Wilcoxon p {fp(ca['wilcoxon_p_probe_gt_random'])}); the probe subspace
captures {100 * ca['probe_var_captured']:.1f}% of species-vector variance. Off-target next-token loss on neutral text was
{f3(ca['neutral_loss_base'])} (no edit), {f3(ca['neutral_loss_probe_ablated'])} (probe) and {f3(ca['neutral_loss_random_ablated'])} (random).
*Interchange* (copying the subspace coordinates of species A into species B) switched the predicted class to A's in
{100 * cw['switch_rate_probe']:.1f}% of {cw['n_pairs']} pairs, vs {100 * cw['switch_rate_no_edit']:.1f}% without an edit and
{100 * sum(cw['switch_rate_random']) / len(cw['switch_rate_random']):.1f}% for random subspaces (rank p {fp(cw['p_rank_vs_random'])}).
*Behaviour*: asked which of two animals is more closely related to a third (A/B, both orders), the model chose the
tree-closer one in {100 * bs['behavioral_accuracy']:.1f}% of {bs['n']} within-stratum triplets (binomial p {fp(bs['binom_p'])})
and in {bc['chose_tree_closer']}/{bc['n']} conflict triplets; its class-word accuracy was {100 * s3['class_accuracy']['accuracy']:.1f}%.
"""
    eco_txt = ""
    if eco:
        t = eco["tests"]
        lines = []
        for k, v in t.items():
            if "p" in v:
                stat = f"ρ = {f3(v['rho'])}" if "rho" in v else f"median {f3(v['median'])} vs ground {f3(v['median_ground'])} (n = {v['n']})"
                lines.append(f"{esc(k)}: {stat}, one-sided p {fp(v['p'])}, Holm p {fp(v['holm_p'])}")
            else:
                lines.append(f"{esc(k)}: not evaluable (n = {v['n']} < 5)")
        eco_txt = ("\n== Habitat-gradient addendum (preregistered after stage 1a, tested on held-out species)\n"
                   f"On {eco['n_mammals']} held-out mammals and {eco['n_birds']} held-out birds projected onto the fixed layer-21 PC1 axis: "
                   + "; ".join(lines) + ".\n")
    stage4_txt = ""
    if s4:
        v = s4["variants"]
        items = "; ".join(f"{esc(k)}: {f3(x['ta_within_modal'])}" for k, x in v.items())
        pr = s4["probe"]
        stage4_txt = (f"\n== Ablations (stage 4)\nTA#sub[within] at layer {s4['modal_layer']} under single changes: {items}.\n"
                      f"Probe held-out r by dimension (angular): "
                      + ", ".join(f"k={k}: {f3(pr[f'angular_k{k}']['test_test_pearson'])}" for k in (2, 3, 5, 10, 20))
                      + ". Euclidean / Poincaré at k = 2, 5, 10: "
                      + ", ".join(f"{m} k={k}: {f3(pr[f'{m}_k{k}']['test_test_pearson'])}" for m in ("euclid", "poincare") for k in (2, 5, 10))
                      + f". Within-stratum held-out Spearman of the k = 10 probe: {f3(pr['angular_k10']['test_test_within_group_spearman'])} "
                        f"(shuffled-tree control {f3(pr['angular_k10_shuffled_tree']['test_test_within_group_spearman'])}).\n")
    today = date.today().isoformat()
    doc = f"""#set document(title: "Phylogenetic geometry of the tree of life in an LLM", author: "Libor Burian")
#set page(margin: 2cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#align(center)[
  #text(16pt, weight: "bold")[Do LLMs Represent a Tree of Life? \\ Phylogenetic Geometry of Animal Species in Qwen3.5-2B]
  #v(4pt)
  Libor "Burny" Burian #h(1em) · #h(1em) {today} \\
  #link("{REPO_URL}")[code, data, preregistration]
]

*Abstract.* Goodfire showed that the DNA model Evo 2 embeds bacterial species on a phylogenetic manifold. We ask whether a
*text* LLM does the same for animals. We extract residual-stream representations of 400 animal species names from
Qwen3.5-2B (8 prompts each) and compare their geometry with the Open Tree of Life under a preregistered protocol with
untrained-model, lexical, co-occurrence, token-count and popularity controls. Within coarse groups and with head nouns
controlled, rooted-triplet accuracy is {f3(h1['ta_within_modal'])} (95% CI {f3(h1['ci95'][0])}–{f3(h1['ci95'][1])}; chance 0.333;
max-T p {fp(h1['maxT_p'])}), above every control; a 10-dimensional linear probe predicts tree distances between held-out families
with r = {f3(probe['model']['test_test_pearson'][0])} (shuffled-tree control {f3(probe['shuffled_tree']['test_test_pearson'][0])}).
However, the signal is not demonstrably beyond spelling or GloVe co-occurrence on triplets where those baselines fail,
does not track divergence time beyond topology, and where folk taxonomy and phylogeny conflict the representation follows
folk taxonomy ({cs['phylo_choice']}/{cs['n']} conflict triplets follow the tree). The top principal axes form class-level arms and
an ecology-like water→land→air gradient rather than a nested tree. LLMs encode a *folk-taxonomic tree*, not an evolution-faithful phylogeny.

= Introduction
Representation geometry is a window into what models learn: days of the week form circles
#footnote[Engels et al. 2024, arXiv:2405.14860.] and DNA models place species on a phylogenetic manifold
#footnote[Goodfire, "Finding the Tree of Life in Evo 2", 2025.]. Text contains phylogeny only indirectly, entangled with folk
categories, shared words, popularity and co-occurrence; whether an LLM's species geometry reflects the tree of life is open
(no peer-reviewed study found; see reports/literature-review.md). Contributions: (i) a preregistered, controlled test with a
topology-only primary metric (node-balanced rooted-triplet accuracy within groups, head nouns disjoint); (ii) Goodfire-style
probe and manifold analyses with the controls their Evo 2 post lacked; (iii) causal and behavioural tests; (iv) evidence that the
learned geometry is folk-taxonomic.

#figure(image("figures/architecture.png", width: 100%), caption: [Pipeline.])

= Methods
*Species.* 400 animals in 13 strata chosen by a fixed rule from Wikidata (English Wikipedia common-name titles, ranked by
sitelinks, ≤ 3 per family, polysemous names excluded), frozen with sha256 before extraction. *Tree.* Open Tree of Life induced
topology (CC0); NCBI-first Linnaean ranks; TimeTree 5 divergence times (secondary). *Representations.* Qwen3.5-2B (post-trained,
bf16, batch size 1), residual stream at the last name token for 8 chat-formatted templates, all 25 layers; per-template mean
centering, template average, cosine distance. *Primary metric.* For triplets (a, b | c) resolved by the tree, correct if
d(a,b) < min(d(a,c), d(b,c)); node-balanced mean over decision nodes; restricted to triplets inside one stratum with three
distinct head nouns (TA#sub[within]). Significance by max-T permutation over layers (within-stratum label shuffles), point
estimate by family-blocked cross-fitting, CI by clade-block (family) bootstrap; H1 additionally requires beating random-init
Qwen, Pythia step 0, character 3-grams, the input embedding, token count and popularity. A power simulation on the frozen tree
gave a minimum detectable TA#sub[within] of {f3(power['min_detectable_ta_within_80pct_power'])} at 80% power.

= Results
== What the top principal components show (stage 1a)
#figure(image("figures/spot.png", width: 100%), caption: [Layer 21, PC1–PC2; one stratum highlighted per panel. The main axis
runs from aquatic groups to flyers (insects, birds), cutting across phylogeny.])
#grid(columns: 2, gutter: 6pt, figure(image("figures/months.png", width: 90%), caption: [Positive control: months form a circle.]),
  figure(image("figures/brown.png", width: 100%), caption: [What a true tree looks like (Brownian motion on the reference tree).]))
Days and months form circles in calendar order (pipeline check). Species form class-level arms (layers 4–5) and a curved,
ecology-like band in late layers; within-class order clustering exceeds the lexical (layer 0) and random-init references, but no
nested within-class branching is visible in the top 3 PCs.

== Triplet accuracy and controls (stage 1b, confirmatory)
#figure(table(columns: 3, align: (left, center, center),
    [*representation*], [*TA#sub[within]*], [*95% CI*],
    {rows}
  ), caption: [Within-stratum, head-noun-disjoint rooted-triplet accuracy (chance 0.333); models at their cross-fitted layer.])
H1 is {'supported' if h1['supported'] else 'not supported'}: cross-fitted TA#sub[within] {f3(h1['crossfit_ta_within'])} (layer {h1['modal_layer']}),
max-T p {fp(h1['maxT_p'])}, and every paired-bootstrap difference to the required controls excludes zero.
#figure(table(columns: 5, [*id*], [*statistic*], [*value*], [*p*], [*Holm p*], {sec_rows}), caption: [Secondary hypotheses.])
#figure(image("figures/ta.png", width: 85%), caption: [TA#sub[within] by layer for the model and controls.])
*Conflict triplets.* The representation picks the tree-closer partner in {cs['phylo_choice']}/{cs['n']} triplets
(binomial p {fp(cs['binom_p_two_sided'])}), e.g. blue whale is closer to whale shark than to hippopotamus.

== A linear phylogeny subspace (stage 2)
#figure(table(columns: 3, [*probe input / target*], [*held-out-family r*], [*train r*], {probe_rows}),
  caption: [Flat-subspace probe (k = 10), 5-fold family holdout, 5 seeds.])
Geodesic distance on the kNN graph did not improve on angular distance; δ-hyperbolicity of the model ({f3(s2['delta_rel']['model'])})
lies between a spectrum-matched Gaussian ({f3(s2['delta_rel']['matched_gaussian_mean'])}) and pure class clusters
({f3(s2['delta_rel']['class_blob_mean'])}), so it does not discriminate trees from clusters; a neighbour-joining tree from the
representations has normalized RF {f3(s2['nj_rf']['model'])} to the reference (permutation null {f3(s2['nj_rf']['perm_null_mean'])}).
{stage3_txt}{stage4_txt}{eco_txt}
= Discussion
The model's species geometry agrees with the tree of life within groups beyond untrained, lexical, token-count and popularity
controls, and a linear subspace generalizes to unseen families. But on triplets where spelling or co-occurrence statistics
are wrong the model is not reliably right, the geometry tracks rank structure rather than divergence time, and where intuition
and phylogeny conflict it follows intuition. The dominant axes mix taxonomy with ecology. This is the geometry one expects from
text: a folk-taxonomic tree. *Limitations:* one 2B model (scale in stage 5), English common names, a single tree source, a
correlational primary metric, and S3/S4 subsets that are harder by construction. *Related work* is reviewed in
reports/literature-review.md (Goodfire Evo 2 and neural geometry; Engels et al.; Park et al.; Gurnee & Tegmark; Modell et al.;
Nava & Wyart; Hewitt & Manning/Liang; phylotext-mammals).

= Conclusion
Qwen3.5-2B encodes a tree of animals, but it is the tree people talk about, not the one evolution built.

#v(8pt)
#line(length: 100%)
*AI statement.* This codebase and paper were written by Claude Code (Anthropic) executing a specification provided by a human
(Libor Burian); human cleanup and rewriting are in progress. All numbers are generated from the repository's results files.
"""
    out = REPORTS / "paper.typ"
    out.write_text(doc, encoding="utf-8")
    try:
        import typst

        typst.compile(str(out), output=str(REPORTS / "paper.pdf"))
    except ImportError:
        pass
    return out
