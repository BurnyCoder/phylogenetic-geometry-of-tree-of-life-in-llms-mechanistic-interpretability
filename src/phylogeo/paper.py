"""Build the manuscript from the same corrected evidence tables as the public report."""

import shutil
import subprocess
from pathlib import Path

from phylogeo.config import REPO_ROOT
from phylogeo.verification.publication import neighbor_examples, section_texts, src

REPORTS = REPO_ROOT / "reports"


def architecture_png(path: Path) -> None:
    """Pipeline diagram (boxes and arrows) in the style of ML papers; non-overlapping layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    H = 0.95
    data, model, stage, ctl = "#eeeeec", "#e8f0fb", "#fdecd9", "#f4f4f2"
    boxes = {  # name: (x, y, width, color, text)
        "wd": (0.1, 3.0, 2.0, data, "Wikidata (QLever)\nfrozen animal panel\ncommon names, NCBI ids"),
        "prompt": (2.6, 3.0, 2.0, model, "8 templates × species\nchat format\n'My favorite animal is the {name}.'"),
        "llm": (5.1, 3.0, 2.0, model, "Qwen3.5-2B (bf16)\nembedding + block readouts\nlast name token"),
        "vec": (7.6, 3.0, 2.0, model, "species vectors\nper-template centering\ncosine distance"),
        "ott": (0.1, 1.5, 2.0, data, "Open Tree of Life topology\n(CC0) + NCBI ranks\n+ TimeTree ages"),
        "ctl": (0.1, 0.15, 2.6, ctl, "Controls: random-init Qwen,\nPythia step0, char-3-grams, GloVe,\ntoken count, popularity, oracles"),
        "s1": (2.95, 0.15, 1.95, stage, "Stage 1: top-3 PCA,\ntriplet accuracy, Mantel,\nconflict triplets"),
        "s2": (5.0, 0.15, 1.95, stage, "Stage 2: 10-d flat probe\n(held-out families),\ngeodesics, δ, NJ tree"),
        "s3": (7.05, 0.15, 1.95, stage, "Stage 3: behaviour;\nablate / swap the\nprobe subspace"),
        "s4": (9.1, 0.15, 2.2, stage, "Stages 4–6: ablations,\nother models (scale,\ntraining), sensitivity"),
    }
    for x, y, w, col, t in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.04", fc=col, ec="0.35", lw=0.8))
        ax.text(x + w / 2, y + H / 2, t, ha="center", va="center", fontsize=7.4)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="0.3"))

    for u, v in (("wd", "prompt"), ("prompt", "llm"), ("llm", "vec")):
        xu, yu, wu = boxes[u][:3]
        arrow((xu + wu, yu + H / 2), (boxes[v][0], yu + H / 2))
    xv, yv, wv = boxes["vec"][:3]
    for st in ("s1", "s2", "s3", "s4"):
        xs, ys, ws = boxes[st][:3]
        arrow((xv + wv / 2, yv), (xs + ws / 2, ys + H))
    arrow((2.1, 1.97), (2.95, 0.75))  # tree -> stages
    arrow((2.7, 0.62), (2.95, 0.62))  # controls -> stages
    ax.text(0.1, 4.2, "Phylogenetic geometry of animal species in an LLM: pipeline", fontsize=9.5, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_paper() -> Path:
    """Convert our source-linked sections to Typst, then compile the verified manuscript."""
    import typst

    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Building the paper requires Pandoc with the Typst writer.")
    sections = section_texts()
    architecture_png(REPORTS / "figures/architecture.png")
    def prose(title, body=None):
        md = f"## {title}\n\n{sections[title] if body is None else body}\n"
        return subprocess.run([pandoc, "-f", "gfm", "-t", "typst", "--wrap=none"], input=md, encoding="utf-8",
                              stdout=subprocess.PIPE, check=True).stdout

    def figure(path, caption, height):
        return f'\n#figure(image("{path}", height: {height}cm, width: 100%, fit: "contain"), caption: [{caption}])\n'

    preamble = r"""#set document(title: "Phylogenetic geometry of animal names in language models", author: "Libor Burian")
#set page(paper: "a4", margin: (x: 1.8cm, y: 1.7cm), numbering: "1")
#set text(size: 10pt, lang: "en")
#set par(leading: 0.6em, justify: false)
#set heading(numbering: none)
#show link: set text(fill: rgb("#244f77"))
#show table: set text(size: 8.5pt)
#set table(inset: (x: 3pt, y: 3pt), stroke: 0.45pt + rgb("#b4c3ce"))
#show figure.caption: set text(size: 8.8pt)
#align(center)[#text(size: 19pt, weight: "bold")[Phylogenetic geometry of animal names in language models]
#linebreak() Libor “Burny” Burian · 26 November 2026]
"""
    readout = (
        "Wikidata and English Wikipedia supplied candidates, common names and sitelinks. Selection used broad taxonomic groups and a family cap. "
        "Open Tree supplied the induced topology. NCBI-first ranks, with Open Tree fallback, annotate the tips. "
        "A deep sister pair can have a longer edge-count path than a shallow outgroup, so the shared ancestor determines the correct pair on each resolved triple. "
        + src("Frozen panel", "data/species.csv") + " " + src("Selection and amendments", "reports/preregistration.md") + "\n\n"
        "The archived Qwen runs used its text decoder in chat format. Token offsets located each name in several sentence frames. "
        "The last name token supplied the primary readout. The embedding layer can already encode a shared final token. "
        "Untrained networks, spelling, the input embedding, token count and sitelinks are the primary controls. Static GloVe embeddings and a rank oracle provide additional comparisons. "
        + src("Prompt construction", "src/phylogeo/prompts.py") + " " + src("Baseline tables", "results/qwen35_2b/SUMMARY.md#stage-1b") + "\n\n"
        "Goodfire’s bacterial genome comparison motivated graph shortest paths and an angular distance probe with clade holdouts. Engels et al. found calendar circles in language-model features. "
        "Modell et al. describe conditions under which distances along a manifold recover conceptual distances. Work on lexical hierarchies and co-occurrence gives a competing explanation for a tree-like arrangement of names. "
        "[Goodfire](https://www.goodfire.com/research/phylogeny-manifold) · [Engels et al.](https://arxiv.org/abs/2405.14860) · "
        "[Modell et al.](https://arxiv.org/abs/2505.18235) · [Park et al.](https://arxiv.org/abs/2406.01506) · "
        "[Nava and Wyart](https://arxiv.org/abs/2605.23821) · " + src("Literature review", "reports/literature-review.md"))
    probe_design = (
        "The angular probe applies a learned linear map to PCA coordinates. A learned scale converts angular separation into predicted tree distance. "
        "The loss combines squared distance error with a reconstruction term. Historical cross-validation assigned whole families to folds. Shuffled targets, random weights and a class-centroid-plus-noise input check different explanations for a good fit. "
        "Within a stratum, probe and raw distances can be compared on the same held-out pairs. " + src("Probe implementation", "src/phylogeo/probes/flat.py") + " "
        + src("Fold design and limitations", "reports/experiments/03-geometry-probes-stage2/experiment.md") + "\n\n"
        "Raw text, per-dimension standardization and within-stratum centering yield estimates close to the main score on this panel. The token readout and common versus scientific name choice give larger changes. "
        "Species subsampling produces more variation than triplet resampling in the archived draws. The template runs record model-only scores. "
        + src("Ablations", "results/qwen35_2b/stage4/results.json") + " " + src("Per-draw scores", "results/qwen35_2b/stage6/results.json"))
    visual_checks = (
        "The early stage compared uncolored and labeled projections with the embedding layer, random weights, calendar words and a simulated Brownian tree. "
        "Broad class clusters can produce a short tree overlay. A calendar loop illustrates what an ordered low-dimensional feature looks like in this pipeline. "
        "The habitat hypothesis arose after inspection of the discovery plots. The subsequent fixed-axis calculation used different species. "
        + src("Discovery record", "reports/experiments/01-pca-shape/experiment.md") + " " + src("Held-out habitat", "reports/experiments/06-habitat-addendum/experiment.md"))
    examples = (
        "The query is the species with the most Wikidata sitelinks in each broad group, chosen by the same rule throughout. "
        "Each column lists its first archived neighbor. The neural representations use their own selected layers. "
        "The tree ranks by deepest shared ancestor, then by edge count. Remaining distance ties use frozen row order. "
        + src("Selection rule", "src/phylogeo/examples.py") + "\n\n" + neighbor_examples(compact=True) + "\n\n"
        "The full output lists three neighbors per query under every representation, including trained Pythia and step-zero Pythia. "
        + src("Full comparison table", "results/RESULTS.md#nearest-neighbor-examples") + " " + src("Original examples", "results/qwen35_2b/examples/neighbors.json"))
    pages = [
        preamble + prose("Abstract") + prose("Question and prior work") + prose("Registration and chronology")
        + figure("figures/architecture.png", "Data, model readouts, reference annotations and evaluation stages in the original experiment. The reanalysis starts with archived readouts.", 6.4),
        prose("Methods") + prose("Hypothesis decisions"),
        prose("Primary geometry and control comparisons") + figure("figures/viz/h1.png", "Primary accuracy and paired family-block intervals. Each dot is a fixed-layer estimate.", 10),
        prose("Secondary hypotheses") + figure("figures/viz/secondary.png", "Trained and untrained estimates on the specified subsets. The spelling-neutral subset is exploratory.", 9),
        prose("Interventions and behavior") + prose("Conflict stimuli")
        + figure("figures/viz/causal.png", "The right panel measures post-edit donor prevalence. The missing paired transition outcomes prevent the registered comparison.", 7),
        prose("Habitat on held-out species") + figure("figures/viz/habitat.png", "Fixed discovery axis projected onto held-out species. The marine comparison is concentrated in two orders.", 8)
        + prose("Exploratory geometry, probes and power"),
        prose("Ablations, sensitivity and model scale") + figure("figures/viz/scale.png", "Archived checkpoint trajectories. Final models differ in several characteristics. The audit identifies duplicate geometry in the scale series.", 7)
        + prose("Limitations and conclusion"),
        prose("Species selection, model readout and related work", readout) + prose("Visual checks", visual_checks)
        + figure("figures/viz/pca_expected.png", "Reference simulation and animal-name projections. Each condition has its own principal components.", 7),
        prose("Probe design, ablations and sensitivity", probe_design)
        + figure("figures/viz/probe.png", "Held-out probe and class-only comparisons. The raw within-stratum comparator uses the same held-out pairs.", 7.3)
        + figure("figures/viz/sensitivity.png", "Variation across analysis choices in the archived sensitivity study.", 7.3),
        prose("Appendix and reproducibility") + figure("figures/viz/view_birds.png", "Bird examples from the archived PCA display. The camera angle was optimized for this view. The reference overlay uses the known topology.", 8)
        + figure("figures/months.png", "Calendar calibration. Months in calendar order, shown in a projection from the original run.", 5),
        prose("Output examples", examples)
        + "\n#v(6pt)\n*AI statement.* Codex wrote the independent verification code and this revision of the paper to Libor Burian’s specification. The original implementation and manuscript attributed their drafting to Claude Code. Human cleanup and rewriting are in progress.\n",
    ]
    source = REPORTS / "paper.typ"
    source.write_text("\n#pagebreak()\n".join(pages), encoding="utf-8")
    typst.compile(str(source), output=str(REPORTS / "paper.pdf"))
    return source
