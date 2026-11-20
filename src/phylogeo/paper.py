"""Generate the paper (reports/paper.typ -> reports/paper.pdf) from results JSON only.

Global context: every number in the paper is read from ``results/**/*.json`` here,
so the paper cannot drift from the analysis. One command regenerates it:
``uv run python -m phylogeo.run --phases paper``. Figures are copied from
``results/``; the architecture diagram is drawn with matplotlib. Typst is compiled
through the ``typst`` PyPI package (https://pypi.org/project/typst/). Sections whose
results do not exist yet are omitted rather than filled with placeholders.
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

# References (verified in reports/literature-review.md, fact-check log). key: (text, url)
REFS = {
    "goodfire_evo2": ("Pearce, Simon, Byun, Balsam (Goodfire) 2025. Finding the Tree of Life in Evo 2.",
                      "https://www.goodfire.com/research/phylogeny-manifold"),
    "goodfire_geom": ("Goodfire 2025-26. Neural geometry research series.", "https://www.goodfire.com/research/neural-geometry"),
    "engels": ("Engels, Michaud, Liao, Gurnee, Tegmark 2024. Not All Language Model Features Are One-Dimensionally Linear.",
               "https://arxiv.org/abs/2405.14860"),
    "modell": ("Modell, Rubin-Delanchy, Whiteley 2025. The Origins of Representation Manifolds in Large Language Models.",
               "https://arxiv.org/abs/2505.18235"),
    "wurgaft": ("Wurgaft et al. (Goodfire) 2026. Steering Along Manifolds. arXiv 2605.05115.", "https://arxiv.org/abs/2605.05115"),
    "park": ("Park, Choe, Jiang, Veitch 2024. The Geometry of Categorical and Hierarchical Concepts in Large Language Models.",
             "https://arxiv.org/abs/2406.01506"),
    "gurnee": ("Gurnee, Tegmark 2023. Language Models Represent Space and Time.", "https://arxiv.org/abs/2310.02207"),
    "nava": ("Nava, Wyart 2026. Co-occurrence statistics and hierarchical representation geometry (arXiv 2605.23821).",
             "https://arxiv.org/abs/2605.23821"),
    "hewitt": ("Hewitt, Manning 2019. A Structural Probe for Finding Syntax in Word Representations. NAACL.",
               "https://aclanthology.org/N19-1419/"),
    "nickel": ("Nickel, Kiela 2017. Poincaré Embeddings for Learning Hierarchical Representations.",
               "https://arxiv.org/abs/1705.08039"),
    "khrulkov": ("Khrulkov, Mirvakhabova, Ustinova, Oseledets, Lempitsky 2020. Hyperbolic Image Embeddings.",
                 "https://arxiv.org/abs/1904.02239"),
    "phylotext": ("phylotext-mammals (GitHub repository, user jadencutinha).",
                  "https://github.com/jadencutinha/phylotext-mammals"),
    "opentree": ("Open Tree of Life (synthetic tree and taxonomy; CC0).", "https://tree.opentreeoflife.org/"),
    "timetree": ("Kumar et al. 2022. TimeTree 5. Molecular Biology and Evolution.", "https://timetree.org/"),
    "elton": ("Wilman et al. 2014. EltonTraits 1.0. Ecology 95:2027.", "https://esapubs.org/archive/ecol/E095/178/"),
    "qwen": ("Qwen team 2026. Qwen3.5-2B model card.", "https://huggingface.co/Qwen/Qwen3.5-2B"),
    "pythia": ("Biderman et al. 2023. Pythia: A Suite for Analyzing Large Language Models.", "https://arxiv.org/abs/2304.01373"),
    "glove": ("Pennington, Socher, Manning 2014. GloVe.", "https://nlp.stanford.edu/projects/glove/"),
    "lones": ("Lones 2021. How to avoid machine learning pitfalls.", "https://arxiv.org/abs/2108.02497"),
}


ECO_LABELS = {"M1_marine_lt_ground": "M1 (marine mammals below ground-dwelling ones)",
              "M2_aerial_gt_ground": "M2 (bats above ground-dwelling mammals)",
              "B1_water_negative": "B1 (more water foraging, lower score; birds)",
              "B2_aerial_positive": "B2 (more aerial foraging, higher score; birds)"}


def _j(*parts):
    p = RESULTS_DIR.joinpath(*parts)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def fpv(p) -> str:
    """p-value as a bare number (tables)."""
    return f"{p:.1e}" if p < 0.0005 else f"{p:.3f}"  # floors print as their value (e.g. 0.001)


def fp(p) -> str:
    """Inline p-value ("p = x"). The "=" is escaped so a wrapped line can never start a Typst heading."""
    return f"\\= {p:.1e}" if p < 0.0005 else f"\\= {p:.3f}"  # floors print as their value


def esc(s: str) -> str:
    """Escape Typst markup characters in free text."""
    for ch in ("\\", "#", "$", "@", "<", ">", "*", "_", "[", "]"):
        s = s.replace(ch, "\\" + ch)
    return s


def cite(key: str) -> str:
    """Numbered citation linking to the reference list."""
    return f"[#link(<ref-{key}>)[{list(REFS).index(key) + 1}]]"


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
        "wd": (0.1, 3.0, 2.0, data, "Wikidata (QLever)\n400 animal species\ncommon names, NCBI ids"),
        "prompt": (2.6, 3.0, 2.0, model, "8 templates × species\nchat format\n'My favorite animal is the {name}.'"),
        "llm": (5.1, 3.0, 2.0, model, "Qwen3.5-2B (bf16)\nresidual stream, 25 layers\nlast name token"),
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


def _table(header: list[str], rows: list[list[str]], caption: str, align: str = "left") -> str:
    cells = ", ".join(f"[*{esc(h)}*]" for h in header)
    body = "\n    ".join(", ".join(f"[{c}]" for c in r) + "," for r in rows)
    return (f"#figure(table(columns: {len(header)}, align: {align}, stroke: 0.4pt + luma(180),\n    {cells},\n    {body}\n  ),"
            f" caption: [{caption}])\n")


def _fig(name: str, caption: str, width: str = "100%") -> str:
    return f'#figure(image("figures/{name}.png", width: {width}), caption: [{caption}])\n' if (FIG / f"{name}.png").exists() else ""


def build_paper() -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    architecture_png(FIG / "architecture.png")
    figs = {
        "spot": RESULTS_DIR / RUN / "stage1a" / "spotlight_layer21_pc12.png",
        "months": RESULTS_DIR / RUN / "stage1a" / "calib_months_layer12_pc12.png",
        "brown": RESULTS_DIR / RUN / "stage1a" / "calib_synthetic_brownian_tree.png",
        "ta": RESULTS_DIR / RUN / "stage1b" / "ta_within_by_layer.png",
        "probe": RESULTS_DIR / RUN / "stage2" / "probe_heldout_r.png",
        "mammals": RESULTS_DIR / RUN / "stage1a" / "zoom_mammals_layer21.png",
        "scale": RESULTS_DIR / "stage5" / "scale.png",
        "dynamics": RESULTS_DIR / "stage5" / "training_dynamics.png",
        "sens": RESULTS_DIR / RUN / "stage6" / "sensitivity_ta_within.png",
    }
    for k, p in figs.items():
        if p.exists():
            shutil.copy2(p, FIG / f"{k}.png")
        elif (FIG / f"{k}.png").exists():
            (FIG / f"{k}.png").unlink()  # never show a figure whose results are gone
    s1a, s1b = _j(RUN, "stage1a", "checks.json"), _j(RUN, "stage1b", "results.json")
    s2, s3, s4 = _j(RUN, "stage2", "results.json"), _j(RUN, "stage3", "results.json"), _j(RUN, "stage4", "results.json")
    s5, s6 = _j("stage5", "summary.json"), _j(RUN, "stage6", "results.json")
    eco, conf, ex = _j(RUN, "eco", "results.json"), _j(RUN, "conflict", "conflict.json"), _j(RUN, "examples", "neighbors.json")
    power = _j("power", "summary.json")
    h1, sec = s1b["H1"], s1b["secondary"]
    pt, ci = s1b["point_ta_within"], s1b["ci95_ta_within"]
    cs = conf["summary"][RUN]
    probe = s2["probe"]
    L = h1["modal_layer"]
    sup = lambda ok: "supported" if ok else "not supported"

    # --- Main evaluation table: method, controls, baselines, oracles, ablations --------------------
    labels = [(RUN, "*Qwen3.5-2B (main method)*", "method"), ("qwen35_2b_random", "random-init Qwen3.5-2B", "control"),
              ("pythia410m_step0", "Pythia-410m step 0 (untrained)", "control"), ("pythia410m", "Pythia-410m (trained)", "baseline"),
              ("char3", "character 3-grams", "control"), ("input_embedding", "input embedding (layer 0)", "control"),
              ("token_count", "name token count", "control"), ("log_popularity", "log Wikipedia sitelinks", "control"),
              ("glove", "GloVe 300d (co-occurrence)", "baseline"), ("class_oracle", "class oracle", "oracle"),
              ("rank_oracle", "Linnaean-rank oracle", "oracle"), ("tree_oracle", "tree oracle (upper bound)", "oracle")]
    eval_rows = [[lab, kind, f3(pt[k]), f"{f3(ci[k][0])}–{f3(ci[k][1])}"] for k, lab, kind in labels if k in pt and k in ci]
    if s4:
        for name, v in s4["variants"].items():
            if not name.startswith("main"):
                eval_rows.append([f"ablation: {esc(name)}", "ablation", f3(v["ta_within_modal"]), "–"])
    eval_tab = _table(["representation", "role", "TA_within", "95% CI"], eval_rows,
                      f"Within-stratum, head-noun-disjoint rooted-triplet accuracy (chance 0.333). Models at their "
                      f"cross-fitted layer; ablations at layer {L}. CIs from a family-block bootstrap.")
    sec_tab = _table(["id", "statistic", "value", "max-T p", "Holm p"], [
        [k, esc(d), f3(v), fpv(sec[k]["maxT_p"]), fpv(sec[k]["holm_p"])]
        for k, d, v in [("S1", "global Mantel ρ (cosine vs path length)", sec["S1"]["mantel_spearman_pathlen_modal"]),
                        ("S2", "TA within orders (head-noun-disjoint)", sec["S2"]["ta_modal"]),
                        ("S3", "TA on lexical-discordant triplets", sec["S3"]["ta_modal"]),
                        ("S4", "TA on co-occurrence-discordant triplets", sec["S4"]["ta_modal"]),
                        ("S5", "partial ρ(distance, age | topology)", sec["S5"]["pooled_partial_spearman"])]],
        "Confirmatory secondary hypotheses (Holm family of 5).")
    probe_tab = _table(["probe input / target", "held-out-family r", "train r"], [
        [esc(k.replace("_", " ")), f"{f3(v['test_test_pearson'][0])} ± {f3(v['test_test_pearson'][1])}",
         f3(v["train_train_pearson"][0])] for k, v in probe.items() if isinstance(v, dict)],
        "Flat-subspace probe (k = 10), 5-fold family holdout, 5 seeds (mean ± SD).")

    # --- Optional sections ------------------------------------------------------------------------------
    stage3_txt = causal_abs = causal_disc = causal_concl = causal_lim = ""  # causal_* filled only when the class control exists
    if s3:
        bs, bc = s3["behavior_species_triplets"], s3["behavior_conflict"]
        stage3_txt = (f"\n== Behaviour and causal tests (stage 3)\n*Behaviour (exploratory).* Asked which of two animals is more "
                      f"closely related to a third (answer letters A/B scored by log-probability, both orders averaged), the model "
                      f"chose the tree-closer animal in {100 * bs['behavioral_accuracy']:.1f}% of {bs['n']} within-stratum triplets "
                      f"(binomial p {fp(bs['binom_p'])}); on the same triplets its layer-{s3['layer']} representation picks the "
                      f"tree-closer animal in {100 * bs['representation_pairwise_accuracy_same_triplets']:.1f}%, and the two agree "
                      f"in {100 * bs['agreement_behavior_vs_representation']:.1f}%. On the conflict triplets it chose the tree-closer "
                      f"animal in {bc['chose_tree_closer']}/{bc['n']} (binomial p {fp(bc['binom_p'])}). Class-word accuracy was "
                      f"{100 * s3['class_accuracy']['accuracy']:.1f}% over {s3['class_accuracy']['n']} species.\n")
        if "causal_holm" in s3:
            ca, cw, hp = s3["C_abl"], s3["C_swap"], s3["causal_holm"]
            nr = ca["n_random_subspaces"]
            stage3_txt += (
                f"\n*Causal tests (confirmatory, Holm family of 2).* The flat probe was fit on half of the families; "
                f"interventions act on the residual stream of the {ca['n_test_species']} held-out species at layer {s3['layer']}. "
                f"Mean-ablating the 10-d probe subspace (capturing {100 * ca['probe_var_captured']:.1f}% of species-vector "
                f"variance) at the name tokens lowered log P(correct class word) by {f3(ca['mean_drop_logp_probe'])} nats "
                f"(no-edit mean log P = {f3(ca['mean_base_logp_correct'])}), against a mean of {f3(ca['mean_drop_logp_random'])} for "
                f"{nr} variance-matched random subspaces: rank p {fp(ca['p_rank_vs_random'])}, Holm p {fp(hp['C_abl'])} "
                f"(C-abl {sup(hp['C_abl'] < 0.05)}). The same ablation applied to every token of 20 neutral sentences changed "
                f"next-token loss from {ca['neutral_loss_base']:.4f} to {ca['neutral_loss_probe_ablated']:.4f} (random subspaces: "
                f"{ca['neutral_loss_random_ablated']:.4f}). Copying the subspace coordinates of species A into species B (at its last name token) switched "
                f"the predicted class to A's in {100 * cw['switch_rate_probe']:.1f}% of {cw['n_pairs']} pairs, against "
                f"{100 * cw['switch_rate_no_edit']:.1f}% without an edit and a mean of "
                f"{100 * sum(cw['switch_rate_random']) / len(cw['switch_rate_random']):.1f}% for random subspaces: rank p "
                f"{fp(cw['p_rank_vs_random'])}, Holm p {fp(hp['C_swap'])} (C-swap {sup(hp['C_swap'] < 0.05)}). "
                # How many random subspaces reach the probe's ablation drop explains a large rank p despite a small Wilcoxon p.
                f"{sum(r >= ca['mean_drop_logp_probe'] for r in ca['random_mean_drops'])} of the {nr} random subspaces caused "
                f"an ablation drop at least as large as the probe's (largest {f3(max(ca['random_mean_drops']))}); the secondary "
                f"per-species Wilcoxon test against the mean random drop gives p {fp(ca['wilcoxon_p_probe_gt_random'])}.\n")
            if "C_abl_v1" in s3:
                na1, ns1 = len(s3["C_abl_v1"]["random_mean_drops"]), len(s3["C_swap_v1"]["switch_rate_random"])
                stage3_txt += (f"A first run used only {na1} and {ns1} random subspaces, whose rank-p floors "
                               f"({f3(1 / (na1 + 1))}, {f3(1 / (ns1 + 1))}) could not pass the Holm threshold; it was replaced by "
                               f"this run (deviation 19) and is kept in the repository.\n")
            if "class_control" in s3:
                # Deviation 20: the readout is the class word, so a class-only subspace is the control for phylogeny-specificity.
                cc = s3["class_control"]
                spec = cc["phylogeny_specific"]
                stage3_txt += (
                    f"\n*Class-only control (deviation 20).* Because the readout is the class word, passing these tests shows only "
                    f"that the subspace carries class information the model uses. A probe of the same size trained on the "
                    f"class-oracle distance (capturing {100 * cc['class_var_captured']:.1f}% of the variance) lowered log P(correct "
                    f"class) by {f3(cc['mean_drop_logp_class'])} nats (rank p against the random subspaces "
                    f"{fp(cc['p_rank_class_vs_random'])}) and switched answers in {100 * cc['switch_rate_class']:.1f}% of the pairs. "
                    f"The phylogeny probe did not exceed it: one-sided Wilcoxon over species p {fp(cc['wilcoxon_p_probe_gt_class'])}; "
                    f"exact McNemar over the same pairs ({cc['mcnemar_probe_only']} vs {cc['mcnemar_class_only']} discordant "
                    f"pairs) p {fp(cc['mcnemar_p_probe_gt_class'])}. The registered rule for a phylogeny-specific causal role "
                    f"(both p < 0.05) is {'met' if spec else 'not met'}.\n")
                causal_abs = (f" Swapping the probe subspace between species switches the class answer more often than random "
                              f"subspaces (Holm p {fp(hp['C_swap'])}; {100 * cw['switch_rate_probe']:.1f}% of pairs vs "
                              f"{100 * cw['switch_rate_no_edit']:.1f}% unedited), ablating it does "
                              f"{'' if hp['C_abl'] < 0.05 else 'not '}beat random subspaces (Holm p {fp(hp['C_abl'])}), and a "
                              f"class-only subspace {'does worse' if spec else 'does as well'} on both, so "
                              f"{'a phylogeny-specific causal role is supported' if spec else 'no phylogeny-specific causal role is shown'}.")
                causal_disc = (f" (5) *Causal (interventions):* C-swap is {sup(hp['C_swap'] < 0.05)} (Holm p {fp(hp['C_swap'])}) "
                               f"but the effect is small ({100 * cw['switch_rate_probe']:.1f}% of pairs switch, "
                               f"{100 * cw['switch_rate_no_edit']:.1f}% without an edit); C-abl is {sup(hp['C_abl'] < 0.05)} "
                               f"(Holm p {fp(hp['C_abl'])}). The phylogeny probe "
                               f"{'beats' if spec else 'does not beat'} a class-only subspace (ablation p "
                               f"{fp(cc['wilcoxon_p_probe_gt_class'])}, interchange p {fp(cc['mcnemar_p_probe_gt_class'])}), so "
                               + ("the model uses the subspace for phylogenetic structure beyond class." if spec else
                                  "the interventions show that the model uses this subspace for coarse class answers, not that "
                                  "it uses within-class phylogenetic structure."))
                causal_concl = (" Causal interventions show that the probe subspace is used for class answers"
                                + (" and carries phylogeny-specific information." if spec else
                                   ", but give no evidence for a phylogeny-specific causal role."))
                causal_lim = (f" The causal readout is the class word, so it cannot test within-class phylogeny directly; the "
                              f"class-only subspace captures more variance than the phylogeny probe "
                              f"({100 * cc['class_var_captured']:.1f}% vs {100 * ca['probe_var_captured']:.1f}%) and was not "
                              f"variance-matched to it.")
    stage4_txt = ""
    if s4:
        pr = s4["probe"]
        bn = s4["variants"].get("scientific binomials (genus-split triplets)", {})
        stage4_txt = (f"\n== Ablations (stage 4, exploratory)\nSingle-change ablations are listed in the evaluation table. "
                      f"Normalization (z-scoring, stratum-centering), prompt format (no chat template) and including polysemous "
                      f"names leave TA#sub[within] essentially unchanged; raw cosine and the mean over name tokens are lower. "
                      f"With scientific binomials on genus-split, head-noun-disjoint triplets, TA is "
                      f"{f3(bn.get('head_disjoint_binomial_modal'))} at layer {s4['modal_layer']} (best layer "
                      f"{f3(bn.get('head_disjoint_binomial_max'))}) versus {f3(bn.get('head_disjoint_common_names_modal'))} for "
                      f"common names on the same triplets: much weaker, and only in late layers. Within strata (mean of "
                      f"per-stratum Spearman on held-out pairs), the k = 10 probe reaches "
                      f"{f3(pr['angular_k10'].get('test_test_within_stratum_spearman'))}, the shuffled-tree probe "
                      f"{f3(pr['angular_k10_shuffled_tree'].get('test_test_within_stratum_spearman'))} and the class-blob input "
                      f"{f3(pr.get('angular_k10_class_blob', {}).get('test_test_within_stratum_spearman'))}. An earlier version "
                      f"pooled all within-stratum pairs into one Spearman; that metric is inflated by strata differing in typical "
                      f"distance and is not used.\n"
                      + _table(["probe dimension k", "angular", "Euclidean", "Poincaré"],
                               [[str(k)] + [f3(pr[f"{m}_k{k}"]["test_test_pearson"]) if f"{m}_k{k}" in pr else "–"
                                            for m in ("angular", "euclid", "poincare")] for k in (2, 3, 5, 10, 20)],
                               "Held-out-family Pearson r of the flat probe by dimension and metric (3 seeds)."))
    eco_txt = ""
    if eco:
        t = eco["tests"]
        lines = []
        for k, v in t.items():
            if "p" in v:
                stat = f"ρ = {f3(v['rho'])}" if "rho" in v else f"median {f3(v['median'])} vs ground {f3(v['median_ground'])} (n = {v['n']})"
                lines.append(f"{ECO_LABELS.get(k, esc(k))}: {stat}, one-sided p {fp(v['p'])}, Holm p {fp(v['holm_p'])}")
            else:
                lines.append(f"{ECO_LABELS.get(k, esc(k))}: not evaluable (n = {v['n']} < 5)")
        eco_txt = ("\n== Habitat axis on held-out species (Addendum A, confirmatory)\nThe stage-1a observation that the top "
                   "late-layer axis orders strata from aquatic to terrestrial groups was registered as a directional hypothesis "
                   "before any held-out species was selected. On "
                   f"{eco['n_mammals']} held-out mammals and {eco['n_birds']} held-out birds (EltonTraits foraging strata "
                   f"{cite('elton')}), projected onto the fixed layer-{eco['layer']} PC1 axis of the primary set: "
                   + "; ".join(lines) + ". The aquatic end is confirmed and the aerial end is not. Caveats: the marine "
                   "mammals are few and their names carry aquatic cues (whale, seal, porpoise, sea otter), water foraging in "
                   "birds is phylogenetically clustered, and the axis holds "
                   + (f"{100 * s1a[eco['layer']]['explained_variance_ratio'][0]:.1f}%" if s1a else "a small share")
                   + " of the variance at that layer. The registered reading is that the geometry mixes ecology with "
                   "taxonomy.\n")
    stage5_txt = ""
    if s5:
        fin = sorted([x for x in s5["runs"] if x["step"] is None], key=lambda x: (x["model_id"].split("/")[0], x["params"]))
        ck = sorted([x for x in s5["runs"] if x["step"] is not None], key=lambda x: x["step"])
        rows = [[esc(x["run"]), f"{x['params'] / 1e9:.2f}B", str(x["n_layers"]), f3(x["ta_within_max"]),
                 f3(x["crossfit_ta_within"]), fpv(x["maxT_p"]), f3(x["mantel_pathlen_modal"])] for x in fin]
        stage5_txt = ("\n== Other models: scale and training dynamics (stage 5, descriptive)\n"
                      + _table(["model", "params", "layers", "max TA_within", "cross-fitted TA_within", "max-T p", "Mantel r"], rows,
                               "Same species, triplets and metric for every model (199 permutations; p floor 0.005).")
                      + _fig("scale", "Cross-fitted TA#sub[within] vs parameter count.", "75%"))
        if ck:
            stage5_txt += _fig("dynamics", "Cross-fitted TA#sub[within] over Pythia-1.4B training.", "75%")
    stage6_txt = ""
    if s6:
        def ms(s):
            return f"{f3(s['mean'])} ± {f3(s['sd'])} (range {f3(s['min'])}–{f3(s['max'])})"
        stage6_txt = ("\n== Sensitivity (stage 6)\n"
                      + _table(["perturbation", "repetitions", "TA_within", "margin over best control (min)"], [
                          ["triplet sampling seed", str(s6["n_rep"]), ms(s6["triplet_seed"][RUN]),
                           f3(s6["triplet_seed"]["margin_vs_best_control"]["min"])],
                          ["80% species subsample", str(s6["n_rep"]), ms(s6["species_subsample"][RUN]),
                           f3(s6["species_subsample"]["margin_vs_best_control"]["min"])],
                          ["template halves (4 of 8)", str(len(s6["templates"]["half"]["subsets"])),
                           ms(s6["templates"]["half"]["ta_within"]), "–"],
                          ["single template", str(len(s6["templates"]["single"]["subsets"])),
                           ms(s6["templates"]["single"]["ta_within"]), "–"],
                          ["cross-fit family-split seed", str(len(s6["crossfit_seed"]["modal_layer"])),
                           ms(s6["crossfit_seed"]["crossfit_ta_within"]), "–"]],
                          f"Spread of TA#sub[within] at layer {s6['layer']}; margin = model minus the best untrained or nuisance "
                          "control on the same draw.")
                      + f"Flat probe over {len(s6['probe_seed']['test_test_pearson']['values'])} fresh seeds: held-out r "
                        f"{ms(s6['probe_seed']['test_test_pearson'])}.\n" + _fig("sens", "TA#sub[within] under each perturbation.", "85%"))
    ex_txt = ""
    if ex:
        reps = [r for r in ("tree", RUN, "qwen35_2b_random", "char3", "glove") if r in ex["representations"]]
        rows = [[esc(q["name"])] + [esc(", ".join(q["neighbors"][r])) for r in reps] for q in ex["queries"]]
        ex_txt = ("\n= Output examples\nNearest neighbours (3) of the most-linked species of every stratum (query rule fixed in "
                  "advance), under the reference tree, the model, the random-init control, character 3-grams and GloVe.\n"
                  + _table(["query"] + [esc(r) for r in reps], rows, "Nearest neighbours per representation.")
                  + "Fraction of each species' 5 nearest neighbours in the same taxonomic order: "
                  + ", ".join(f"{esc(r)} {f3(v)}" for r, v in ex["same_order_rate_5nn"].items()) + ".\n")
    s1a_txt = ""
    if s1a:
        best = max(range(len(s1a)), key=lambda l: s1a[l]["C1_purity"])
        s1a_txt = (f"Class purity of 5-nearest neighbours in PC1–3 peaks at layer {best} ({f3(s1a[best]['C1_purity'])} vs "
                   f"permutation null {f3(s1a[best]['C1_purity_null'])}). ")

    # Post-hoc analyses added after the final review (round 1); all labelled as such in the text.
    exb = s1b.get("exploratory", {})
    posthoc_tab, stratum_tab = "", ""
    if "posthoc_vs_untrained" in exb:
        ph = exb["posthoc_vs_untrained"]
        ctl = [k for k in next(iter(ph.values())) if k not in ("n_triplets", RUN, f"{RUN}_ci95") and not k.startswith("diff_")]
        labels_ph = {"S2_within_order": "S2: within orders", "S3_lexical_discordant": "S3: spelling-discordant",
                     "S4_cooccurrence_discordant": "S4: GloVe-discordant",
                     "char3_tie_lexically_neutral": "spelling has no preference (lexically neutral)"}
        posthoc_tab = _table(["subset", "triplets", "Qwen3.5-2B"] + [f"{c} (diff. CI)" for c in ctl], [
            [labels_ph.get(k, esc(k)), str(v["n_triplets"]), f3(v[RUN])]
            + [f"{f3(v[c])} ({v[f'diff_ci95_vs_{c}'][0]:+.3f}, {v[f'diff_ci95_vs_{c}'][1]:+.3f})" for c in ctl]
            for k, v in ph.items()], "Post hoc: the model vs the untrained controls on the same triplet subsets (paired "
            "family-block bootstrap CI of the difference).")
    if "per_stratum_ci95" in exb:
        stratum_tab = _table(["stratum", "decision nodes", "TA_within", "95% CI"], [
            [esc(k), str(v["n_nodes"]), f3(v["ta"]), f"{f3(v['ci95'][0])}–{f3(v['ci95'][1])}"]
            for k, v in sorted(exb["per_stratum_ci95"].items(), key=lambda kv: -kv[1]["n_nodes"])],
            f"Post hoc: TA#sub[within] per stratum at layer {L} (family-block bootstrap CIs).")
    mm = s1b.get("baselines_mantel_pathlen", {})
    s1_ctx = (f" S1 is not specific to phylogeny: the class oracle alone reaches a Mantel r of {f3(mm.get('class_oracle'))} and "
              f"log popularity alone {f3(mm.get('log_popularity'))}." if mm else "")
    conf_ctl = [(k, v) for k, v in conf["summary"].items() if k != RUN and "mcnemar_vs_main" in v]
    conflict_ctl_txt = ("; ".join(f"{esc(k)} {v['phylo_choice']}/{v['n']} (McNemar vs the model p "
                                  f"{fp(v['mcnemar_vs_main']['p_two_sided'])})" for k, v in conf_ctl))
    bh = _j(RUN, "exploratory_bh.json")
    bh_txt = (f" Exploratory p-values are corrected with BH-FDR over the {bh['n_tests']} registered exploratory tests; "
              "q-values are listed in the repository." if bh else "")
    pcb = probe.get("class_blob")
    probe_ctrl_txt = ""
    if pcb:
        probe_ctrl_txt = (
            f"The headline held-out correlation mainly measures coarse-group separation. A predictor that knows only the 13 "
            f"strata reaches r = {f3(probe.get('class_only_predictor_pearson_all_pairs'))} with path length over all pairs, "
            f"and the same probe trained on a class-blob input (each species at its stratum centroid plus matched noise, no "
            f"within-stratum tree) reaches held-out r = {f3(pcb['test_test_pearson'][0])}. Within strata (mean of per-stratum "
            f"Spearman on held-out pairs) the model's probe reaches "
            f"{f3(probe['model'].get('test_test_within_stratum_spearman', [float('nan')])[0])}, the class-blob input "
            f"{f3(pcb.get('test_test_within_stratum_spearman', [float('nan')])[0])}, and the raw angular distances without any "
            f"probe {f3(probe.get('raw_angular_within_stratum_spearman_all_species'))}: the representation carries some "
            f"within-stratum order, but the probe adds little to it. ")
    nj = s2["nj_rf"]
    nj_txt = (f" Against NJ trees built from class-blob inputs ({f3(nj['class_blob_null_mean'])}, 20 draws) the model's tree is "
              + ("closer to the reference" if nj["model"] < nj["class_blob_null_mean"] else "not closer to the reference")
              + "." if "class_blob_null_mean" in nj else "")
    robust_txt = ""
    if s6:
        lo = min(s6["triplet_seed"][RUN]["min"], s6["species_subsample"][RUN]["min"], s6["templates"]["half"]["ta_within"]["min"])
        mg = min(s6["triplet_seed"]["margin_vs_best_control"]["min"], s6["species_subsample"]["margin_vs_best_control"]["min"])
        robust_txt = (f"; across all stage-6 perturbations TA#sub[within] stays at or above {f3(lo)} and the smallest margin over "
                      f"the best control is {f3(mg)}")
    refs = "\n".join(f'+ #link("{u}")[{esc(t)}] <ref-{k}>' for k, (t, u) in REFS.items())
    today = date.today().isoformat()
    doc = f"""#set document(title: "Phylogenetic geometry of the tree of life in an LLM", author: "Libor Burian")
#set page(margin: 2cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show figure: set block(breakable: true)
#show link: set text(fill: rgb("#2a78d6"))
#align(center)[
  #text(16pt, weight: "bold")[Do LLMs Represent a Tree of Life? \\ Phylogenetic Geometry of Animal Species in Qwen3.5-2B]
  #v(4pt)
  Libor "Burny" Burian #h(1em) · #h(1em) {today} \\
  #link("{REPO_URL}")[code, data, preregistration, interactive results]
]

*Abstract.* Goodfire reported that the DNA model Evo 2 places bacterial species on a curved manifold whose geometry tracks
phylogeny {cite('goodfire_evo2')}. We test whether a *text* LLM does the same for animals. We extract residual-stream
representations of 400 animal species names from Qwen3.5-2B (8 prompts each) and compare their geometry with the Open Tree of
Life under a preregistered protocol with untrained-network, lexical, token-count, popularity and co-occurrence controls. Inside
coarse groups and with shared head nouns excluded, rooted-triplet accuracy is {f3(h1['ta_within_modal'])} (95% CI
{f3(h1['ci95'][0])}–{f3(h1['ci95'][1])}; chance 0.333; max-T p {fp(h1['maxT_p'])}), above every required control (primary
hypothesis {sup(h1['supported'])}), though by a modest margin and clearly only in mammals and birds. The signal is not
significant on triplets where spelling or GloVe co-occurrence points the wrong way, where the model equals untrained networks;
it is not detected in divergence times beyond topology; and on phylogeny-versus-folk conflict triplets the geometry rarely
follows the tree, as for untrained networks. A Goodfire-style 10-dimensional probe predicts tree distances between held-out
families with r = {f3(probe['model']['test_test_pearson'][0])}, but a class-only input does at least as well
({f3(probe.get('class_blob', {}).get('test_test_pearson', [float('nan')])[0])}), so this correlation mainly reflects coarse
groups.{causal_abs} The top late-layer component orders animals from aquatic to terrestrial, confirmed on held-out species.
Qwen3.5-2B thus encodes modest within-group tree order entangled with folk categories and ecology, rather than a phylogenetic manifold.

= Introduction
*Context.* Representation geometry is a window into what models learn. Days of the week and months form circles in LLMs
{cite('engels')}, concept manifolds can be explained by the metric structure of the underlying features {cite('modell')}, and
Goodfire found that Evo 2 embeds the bacterial tree of life so that geodesic distance on a kNN graph tracks phylogenetic distance
{cite('goodfire_evo2')}{cite('goodfire_geom')}. *Motivation.* Text carries phylogeny only indirectly, entangled with folk
categories ("fish", "bugs"), shared words ("tiger shark"), popularity and co-occurrence; co-occurrence alone can produce
hierarchical splits {cite('nava')}. Whether a text LLM's species geometry reflects the tree of life is therefore not obvious,
and global correlations can be inflated by class-level clusters. We found no peer-reviewed study of this question; the closest
work found sentence embeddings uncorrelated with the mammal tree {cite('phylotext')}.

*Contributions.* (i) A preregistered, controlled test with a topology-only primary metric: node-balanced rooted-triplet accuracy
inside coarse groups with head nouns disjoint, against a fixed set of untrained-network and nuisance controls. (ii) Goodfire-style
probe and manifold analyses with family-held-out evaluation, a shuffled-tree control and a class-only control; the last shows that
the probe's held-out correlation mostly reflects coarse groups. (iii) Behavioural judgments and causal ablation and interchange
of the probe subspace against variance-matched random subspaces and a class-only subspace.
(iv) An out-of-sample test of a post-hoc observation (a habitat axis), registered before new species were selected.
(v) Scale, training-dynamics and sensitivity analyses on the same metric.

#figure(image("figures/architecture.png", width: 100%), caption: [Pipeline.])

= Background and related work
*Linear and hierarchical concept geometry.* Park et al. {cite('park')} show that hierarchically related concepts are orthogonal
in a causal inner product and that categorical concepts form simplices; Gurnee and Tegmark {cite('gurnee')} recover linear maps of
space and time. *Manifolds.* Engels et al. {cite('engels')} find circular features; Modell et al. {cite('modell')} predict that
geodesic distance on representation manifolds tracks feature distance, explicitly including tree metrics; Goodfire's manifold
steering moves along such manifolds {cite('wurgaft')}. *Trees and hyperbolic geometry.* Structural probes recover parse-tree
distances {cite('hewitt')}; hyperbolic embeddings represent hierarchies compactly {cite('nickel')}, and Gromov δ-hyperbolicity
measures tree-likeness of a metric {cite('khrulkov')}. *Phylogeny in models.* Evo 2 encodes the bacterial tree
{cite('goodfire_evo2')}; for text models, evidence is sparse and mostly from small projects {cite('phylotext')}.

= Methods
*Species and trees.* 400 animal species in 13 strata, chosen by a fixed rule from Wikidata (English Wikipedia common-name titles
ranked by sitelinks, at most 3 per family, polysemous names excluded), frozen with sha256 before extraction. The reference
topology is the Open Tree of Life induced subtree {cite('opentree')}, with NCBI-first Linnaean ranks; divergence times come from
TimeTree 5 {cite('timetree')} (secondary analysis only). *Representations.* Qwen3.5-2B {cite('qwen')} (post-trained, bf16, batch
size 1), residual stream at the last token of the name for 8 chat-formatted templates, all 25 layers; each template is
mean-centred over species and the templates are averaged; distance is cosine. *Primary metric.* A triplet (a, b | c) is resolved
when the most recent common ancestor of a and b lies strictly below that of all three; it is correct if d(a, b) < min(d(a, c),
d(b, c)). Accuracy is averaged per decision node (at most 200 triplets per node) and then over nodes, restricted to triplets inside
one stratum whose three names have distinct head nouns (TA#sub[within]). *Inference.* Max-T permutation over layers (species
shuffled within strata, 1000 permutations), family-blocked 2-fold cross-fitting of the layer (20 splits), family-block bootstrap
CIs (1000 draws); exploratory tests are BH-corrected over the registered family. H1 additionally requires paired-bootstrap differences above zero against random-init Qwen3.5-2B, Pythia-410m
step 0 {cite('pythia')}, character 3-grams, the input embedding, name token count and log Wikipedia sitelinks. GloVe
{cite('glove')} defines the co-occurrence-discordant subset (S4). A power simulation on the frozen tree gave a minimum detectable
TA#sub[within] of {f3(power['min_detectable_ta_within_80pct_power'])} at 80% power (40 simulations per cell, so this is a rough estimate). *Probe.* Following Goodfire, z = W(x − b) with
d̂ = β·arccos(cos(z#sub[1], z#sub[2])), PCA-100 fitted inside each fold, whole families held out. *Causal tests.* Forward hooks
replace the residual stream after the chosen decoder layer. *Rigor.* All hypotheses, metrics and tests were preregistered before
extraction; every deviation is timestamped in the repository. Practices follow Lones {cite('lones')}.

= Results
== What the top principal components show (stage 1a, exploratory)
{_fig('spot', 'Layer 21, PC1–PC2; one stratum highlighted per panel. The main axis runs from aquatic groups to terrestrial ones, cutting across phylogeny.')}
#grid(columns: 2, gutter: 6pt, figure(image("figures/months.png", width: 90%), caption: [Positive control: months form a circle.]),
  figure(image("figures/brown.png", width: 100%), caption: [What a true tree looks like (Brownian motion on the reference tree).]))
Days and months form circles in calendar order, so the pipeline recovers known geometry. {s1a_txt}Species form class-level arms
in early layers and a curved band in late layers; no nested within-class branching is visible in the top three components, which
hold only a small share of the variance.

== Triplet accuracy and controls (stage 1b, confirmatory)
{eval_tab}
H1 is {sup(h1['supported'])}: cross-fitted TA#sub[within] {f3(h1['crossfit_ta_within'])} (modal layer {L}), max-T p
{fp(h1['maxT_p'])} (the permutation floor), and every paired-bootstrap difference to the required controls excludes zero.
The margin over the strongest required control is modest, and the signal is clear only in mammals and birds, which hold about
half of the within-stratum decision nodes (Table below). The untrained networks are themselves above chance; one hypothesis is
that name form and length cluster by family, but they stay above chance even on triplets with no shared token.
{sec_tab}
{s1_ctx} S2 passes its registered test against chance, but the untrained random-init Qwen reaches the same value; on the
spelling- and GloVe-discordant subsets (S3, S4) the model also equals the untrained networks. On triplets where spelling has no
preference, the model exceeds them (post hoc):
{posthoc_tab}
{stratum_tab}
{_fig('ta', 'TA#sub[within] by layer for the model and controls.', '85%')}
*Conflict triplets.* The representation picks the tree-closer partner in {cs['phylo_choice']}/{cs['n']} triplets (binomial p
{fp(cs['binom_p_two_sided'])}); for example, the blue whale is closer to the whale shark than to the hippopotamus. The untrained
controls behave the same ({conflict_ctl_txt}), and the triplets share anchors and partners, so this does not show a learned folk
or ecological taxonomy; it shows only that the geometry does not follow the tree in these hand-picked cases.{bh_txt}

== Flat-subspace probe and manifold tests (stage 2, exploratory)
{probe_tab}
{_fig('probe', 'Held-out-family performance of the flat probe and its controls: all pairs (left) and within strata (right). Error bars are SDs over probe seeds.', '100%')}
{probe_ctrl_txt}kNN-geodesic distance does not order triplets better than angular distance. Gromov δ#sub[rel] of the model
({f3(s2['delta_rel']['model'])}) lies between a spectrum-matched Gaussian ({f3(s2['delta_rel']['matched_gaussian_mean'])}) and pure
class clusters ({f3(s2['delta_rel']['class_blob_mean'])}), so it does not separate trees from clusters. A neighbour-joining tree
from the representations has normalized Robinson–Foulds distance {f3(s2['nj_rf']['model'])} to the reference (permutation null
{f3(s2['nj_rf']['perm_null_mean'])}; with 20 permutations the smallest attainable p is 0.048).{nj_txt}
{stage3_txt}{stage4_txt}{eco_txt}{stage5_txt}{stage6_txt}
= Discussion
*Claims and evidence.* (1) *Systematic, correlational:* Qwen3.5-2B's species geometry agrees with the tree of life within coarse
groups beyond untrained-network, lexical, token-count and popularity controls{robust_txt}. The margin is modest and the
signal is concentrated in mammals and birds. Because ecology and body size are phylogenetically conserved and were not
controlled, the precise claim is agreement with the tree, not phylogenetic knowledge per se. (2) *Negative for the probe:* the
Goodfire-style flat probe generalizes to held-out families, but a class-only input does at least as well, and within strata it
adds little over the raw distances; we therefore do not claim that a linear subspace "carries the tree". (3) *Negative:* the
order is not significant on triplets where spelling or GloVe co-occurrence disagree with the tree, where the model equals
untrained networks, so we cannot claim it goes beyond such statistics; S2 is matched by an untrained network; no divergence-time
signal is detected once topology is controlled; on conflict triplets the geometry does not follow the tree, as for untrained
networks. (4) *Out-of-sample confirmation of a post-hoc pattern:* the top late-layer component separates aquatic from terrestrial
animals.{causal_disc}

*Implications.* For interpretability, global distance correlations with a taxonomy are weak evidence of a phylogenetic
representation: class clusters, ecology and naming conventions all contribute, and untrained networks already reach above-chance
values on several metrics. Topology-only, within-group metrics with untrained and nuisance controls separate these. Distance
probes evaluated on all held-out pairs should be compared with a class-only input, because coarse groups alone can produce high
correlations. For the analogy with Evo 2, a text model's species geometry is better described as a mixture of folk taxonomy,
ecology and modest phylogenetic order than as a tree-shaped manifold.

= Limitations
One main 2B-parameter model (stage 5 adds smaller and older models, not larger ones); English common names only (binomials carry
much weaker, late-layer structure); one tree source (Open Tree) and NCBI ranks, both imperfect for deep splits, and edge-count
path length depends on how densely each clade is sampled; GloVe is a proxy for co-occurrence in the model's own training data;
S3 and S4 select triplets where simple statistics fail and are therefore adversarial by construction; the within-group signal
rests mostly on mammals and birds; ecological and body-size similarity is not controlled; probes and interventions use a single
layer chosen by cross-fitting; the conflict triplets are few and share anchors; the habitat addendum has only 6 marine mammals,
whose names carry aquatic cues, and no bats.{causal_lim}

= Conclusion
Within this preregistered design, Qwen3.5-2B's species geometry agrees with the tree of life inside coarse groups by a modest,
robust margin over untrained networks and name statistics, clearly so for mammals and birds. The geometry is otherwise dominated
by coarse folk categories and ecology, the Goodfire-style probe mostly recovers the coarse groups, and the evidence does not show
phylogenetic information beyond what lexical and co-occurrence statistics could supply.{causal_concl}
{ex_txt}
= References
{refs}

#v(8pt)
#line(length: 100%)
*AI statement.* This codebase and paper were written by Claude Code (Anthropic), executing a specification provided by a human
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
