"""Generate the hypothesis ledger, numerical tables and independent audit from versioned result files."""

import csv
import json
from pathlib import Path

from phylogeo.config import DATA_REPO_DIR, REPO_ROOT, RESULTS_DIR
from .statistics import MAIN, CONTROLS, OUT, holm_family, read_json, save

BASE = "https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability"
BLOB = BASE + "/blob/main/"
REVIEW = "reports/reviews/2026-11-26-scientific-audit.md"
NAMES = {MAIN: "Qwen3.5-2B", CONTROLS[0]: "Random Qwen", CONTROLS[1]: "Pythia-410m step 0",
         "char3": "Character trigrams", "token_count": "Name token count", "log_popularity": "Log sitelinks",
         "input_embedding": "Input embeddings", "glove": "GloVe", "pythia410m": "Trained Pythia-410m"}
SECONDARY = {"S1": "Global edge-count association", "S2": "Family/genus resolution within order",
             "S3": "Unique wrong spelling choice", "S4": "Unique wrong GloVe choice", "S5": "Age given rank and MRCA depth"}


def src(name, file):
    return f"[{name}]({BLOB}{file})"


def evidence(name):
    return src("Frozen-data calculation", f"results/scientific-verification/{name}.json")


def prob(x):
    return "—" if x is None else (f"{x:.3g}" if x < .001 else f"{x:.4f}")


def num(x):
    return f"{x:.3f}"


def interval(q):
    return f"[{q[0]:.3f}, {q[1]:.3f}]"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def assemble():
    g, d, o = [read_json(OUT / (key + ".json")) for key in ("geometry", "dating", "outcomes")]
    model, family = g["models"][MAIN], g["family"]
    L = model["layer"]
    ps = {k: g["secondary"][k]["p"] for k in ("S1", "S2", "S3", "S4")}
    ps["S5"] = d["partial"]["p_global_max"]
    adj = holm_family(ps)
    ca, eco = o["causal"]["expanded"], o["habitat"]
    required = [k for k in family if k != MAIN]
    h1 = (model["p"] < .05 and family[MAIN]["ci"][0] > 1/3 and
          all(family[k]["difference_ci"][0] > 0 for k in required) and len(required) == 6)
    check = lambda passed: "Criterion met" if passed else "Criterion unmet"
    ledger = [["H1", "Within-group rooted triples", f"{num(model['crossfit'])} cross-fitted",
               prob(model["p"]), check(h1) + " with family blocks"]]
    for k, label in SECONDARY.items():
        value = g["secondary"][k]["modal"] if k != "S5" else d["partial"]["by_layer"][L]
        ledger.append([k, label, f"{num(value)} at layer {L}", prob(adj[k]), check(adj[k] < .05)])
    ledger.extend([
        ["C-abl", "Candidate log-probability drop", f"{num(ca['ablation_probe'])} nats", prob(ca["registered_holm"]["ablation"]),
         "Criterion unmet (Holm upper bound)"],
        ["C-swap", "New donor-class answers", f"{ca['new_donor_bounds'][0]}–{ca['new_donor_bounds'][1]} / {ca['pairs']}",
         "—", "Paired comparison unevaluable"]])
    for key, label in (("M1", "Marine versus ground mammals"), ("M2", "Aerial versus ground mammals"),
                       ("B1", "Bird water foraging"), ("B2", "Bird aerial foraging")):
        effect = (f"{eco['marine']} marine / {eco['ground']} ground" if key == "M1" else
                  f"{eco['aerial']} aerial" if key == "M2" else f"ρ = {num(eco['rho'][key])}")
        v = eco["holm"][key]
        ledger.append([key, label, effect, prob(v), "Unevaluable" if v is None else check(v < .05)])
    return g, d, o, {"layer": L, "H1": h1, "raw_p": ps, "secondary_holm": adj, "hypotheses": ledger}


def headline():
    *_, decisions = assemble()
    return table(["ID", "Hypothesis", "Estimate", "Test p", "Assessment"], decisions["hypotheses"]) + (
        "\n\nH1 uses the layer maximum. Secondary and habitat rows report Holm-adjusted p-values. "
        "Secondary effects are displayed at the H1 layer; their tests cover all layers. C-swap is bounded by the "
        "observed marginals. " + src("Audit, evidence and interpretation", REVIEW) + "\n")


def controls_table(g):
    return table(["Representation", "TA", "Family 95% CI", "Paired Δ, families", "Paired Δ, orders"],
                 [[NAMES[k], num(v["ta"]), interval(v["ci"]), interval(v["difference_ci"]) if k != MAIN else "—",
                   interval(g["order"][k]["difference_ci"]) if k != MAIN else "—"] for k, v in g["family"].items()])


def secondary_table(g, d, decisions):
    return table(["ID", "Triplets or pairs", "H1-layer effect", "Layer-max p", "Holm p"],
                 [[k, g["secondary"][k].get("triplets", "All pairs") if k != "S5" else d["dated_pairs_in_bins"],
                   num(g["secondary"][k]["modal"] if k != "S5" else d["partial"]["by_layer"][decisions["layer"]]),
                   prob(decisions["raw_p"][k]), prob(decisions["secondary_holm"][k])] for k in SECONDARY])


def primary_details(g, archived):
    """Show the component checks alongside the decision. Legacy-only estimates carry their scoring rule."""
    model_rows = [[NAMES[k], v["layer"], num(v["crossfit"]), prob(v["p"])] for k, v in g["models"].items()]
    subset_rows = []
    subsets = [(k, g["secondary"][k]) for k in ("S2", "S2_broad", "S3", "S4")]
    subsets.append(("No shared character trigram", g["no_shared_trigram"]))
    for label, item in subsets:
        comparison = item["comparison"] if label == "No shared character trigram" else item["comparisons"]
        subset_rows.append([label, item["triplets"], num(comparison[MAIN]["ta"]),
                            *[interval(comparison[k]["difference_ci"]) for k in CONTROLS]])
    groups = archived["exploratory"]["per_stratum_ci95"]
    group_rows = [[group, v["n_nodes"], num(v["ta"]), interval(v["ci95"])] for group, v in groups.items()]
    other = {"pythia410m": "Trained Pythia-410m", "glove": "GloVe", "glove_known_names_only": "GloVe, covered names",
             "class_oracle": "Broad group only", "rank_oracle": "Shared taxonomic ranks", "tree_oracle": "Reference topology"}
    baseline_rows = [[label, num(archived["point_ta_within"][key])] for key, label in other.items()]
    correlations = archived["baselines_mantel_pathlen"]
    return "\n\n".join([
        "Each network selects its own layer. The permutation p compares its maximum across layers with the null maxima.",
        table(["Network", "Modal layer", "Cross-fitted TA", "Layer-max p"], model_rows),
        "The paired intervals below resample families. S2 uses the family/genus condition. S2_broad includes all eligible within-order triples. The spelling-neutral row is exploratory.",
        table(["Subset", "Triples", "Qwen TA", "Δ vs random Qwen, 95% CI", "Δ vs Pythia step 0, 95% CI"], subset_rows),
        "Archived descriptive breakdown at the main model’s selected layer. Groups with few decision nodes can have wide intervals. These marginal family intervals have no multiplicity adjustment across groups.",
        table(["Group", "Decision nodes", "TA", "Family 95% CI"], group_rows),
        "Additional archived comparators, using the original fractional credit for tied minima. The tree oracle orders relatives by shared-ancestor depth.",
        table(["Reference or baseline", "Within-group TA"], baseline_rows),
        f"Global edge-count Spearman correlation is {num(correlations['class_oracle'])} for a group-only distance and {num(correlations['log_popularity'])} for log sitelinks. Both can contribute to global agreement. "
        + evidence("geometry") + " " + src("Original breakdowns", "results/qwen35_2b/stage1b/results.json"),
    ])


def neighbor_examples(*, compact=False):
    """Read the fixed queries and recorded neighbors directly from the original output file."""
    archived = read_json(RESULTS_DIR / MAIN / "examples/neighbors.json")
    representations = ["tree", MAIN, CONTROLS[0], "char3", "glove"] if compact else ["tree", *[r for r in archived["representations"] if r != "tree"]]
    labels = {"tree": "Reference tree", **NAMES}
    rows = []
    for query in archived["queries"]:
        neighbors = query["neighbors"]
        cells = [neighbors[r][0] if compact else ", ".join(neighbors[r]) for r in representations]
        rows.append([query["name"], *cells])
    return table(["Query", *[labels[r] for r in representations]], rows)


def neighbor_details():
    """Explain how to read the archived query panel, alongside its full table."""
    return ("Qwen3.5-2B and comparators. The query in each broad group has the most Wikidata sitelinks. "
            "Each cell lists its three nearest other species. Neural representations use their own selected layers. "
            "The reference orders neighbors by deepest common ancestor, breaking ties by edge count and then frozen row order. "
            + src("Recorded neighbors", f"results/{MAIN}/examples/neighbors.json") + "\n\n" + neighbor_examples())


def conflict_table(o):
    return table(["Representation", "Tree closer", "Tied", "Other closer", "Missing"],
                 [[NAMES[k], v["win"], v["tie"], v["loss"], v["unscorable"]] for k, v in o["conflict"]["summary"].items()])


def conflict_items(o):
    """Show every frozen challenge with its recorded category and tie-aware choices."""
    with (DATA_REPO_DIR / "conflict_triplets.csv").open(encoding="utf-8", newline="") as stream:
        items = {int(r["id"]): r for r in csv.DictReader(stream)}
    representations = list(o["conflict"]["summary"])
    labels = {1: "Tree", -1: "Other", 0: "Tie", None: "Missing"}
    rows = []
    for r in o["conflict"]["rows"]:
        item = items[r["id"]]
        details = [r["closer"], r["other"]] if r["resolved"] else ["Unresolved", "Unresolved"]
        choices = [labels[r["choices"][k]] for k in representations] if r["resolved"] else ["—"] * len(representations)
        rows.append([r["id"], item["anchor"], *details, item["conflict_type"], *choices])
    return table(["Item", "Anchor", "Tree-closer animal", "Alternative", "Conflict type",
                  *[NAMES[k] for k in representations]], rows)


def causal_table(o):
    return table(["Run", "Random spaces (ablation / swap)", "Donor before / after", "New donor bounds", "Donor prevalence rank p"],
                 [[name, f"{r['random_ablations']} / {r['random_interchanges']}", f"{r['before']} / {r['after']}",
                   f"{r['new_donor_bounds'][0]}–{r['new_donor_bounds'][1]}", prob(r["donor_prevalence_rank"])]
                  for name, r in zip(("Original", "Expanded"), (o["causal"]["_v1"], o["causal"]["expanded"]))])


def habitat_table(o):
    h = o["habitat"]
    return table(["Prediction", "Species-level p", "Holm (four slots)", "Within-order sensitivity, Holm"],
                 [[k, prob(p), prob(h["holm"][k]), prob(h["order_conditional_holm"][k])] for k, p in h["p"].items()])


def scale_table():
    rows = read_json(OUT / "scale.json")
    return table(["Archived run", "Nominal parameters", "Cross-fitted TA", "Modal layer", "Layer-max p"],
                 [[k, f"{r['nominal_params'] / 1e9:.2f}B", num(r["crossfit"]), r["layer"], prob(r["p"])]
                  for k, r in rows.items()]) + "\n\n" + evidence("scale")


def section_texts():
    g, d, o, decisions = assemble()
    L, h, family = decisions["layer"], g["models"][MAIN], g["family"]
    strongest = max((k for k in family if k != MAIN), key=lambda k: family[k]["ta"])
    fine, broad = (g["secondary"][k] for k in ("S2", "S2_broad"))
    n = g["frozen"]["species"]["n_rows"]
    e, c, ex = o["habitat"], o["causal"]["expanded"], o["exploratory"]
    probe, p = ex["probe"], d["partial"]
    graph = read_json(OUT / "graph.json")
    graph0, graph_l = graph["by_layer"][0], graph["by_layer"][L]
    captured = o["subspace_variance"]
    actual = g["family"][strongest]
    sources = " ".join(src(k.title(), f"results/scientific-verification/{k}.json") for k in ("geometry", "dating", "outcomes"))
    crit = "met" if decisions["H1"] else "did not meet"
    summary = (f"The {n} English animal names in this panel show within-group agreement with the frozen Open Tree "
               f"topology. Family-cross-fitted triplet accuracy was {num(h['crossfit'])}, against one-third chance. "
               f"At layer {L}, accuracy was {num(family[MAIN]['ta'])} with a family-block 95% interval "
               f"{interval(family[MAIN]['ci'])}. The primary result {crit} its family-block decision rule. "
               f"The intervention archive establishes {c['after']} donor-class answers after editing and {c['before']} before "
               f"editing in {c['pairs']} trials. Its registered comparison of genuine answer switches remains unevaluable. {sources}")
    times = (f"The partial Spearman value at layer {L} is {num(p['by_layer'][L])}. Its fixed-layer p is "
             f"{prob(p['p_fixed_by_layer'][L])}. The global two-sided maximum occurs at layer {p['best_absolute_layer']} "
             f"(p {prob(p['p_global_max'])}, Holm {prob(decisions['secondary_holm']['S5'])}). For a claim specifically "
             f"about layer {L}, the layer-adjusted p is {prob(p['p_adjusted_by_layer'][L])}. The original unpartialled "
             f"pooled rank statistic at this layer is {num(d['raw']['by_layer'][L])} (global max p "
             f"{prob(d['raw']['p_global_max'])}). The amended statistic adjusts for depth of the common ancestor within "
             f"rank bins using {d['dated_pairs_in_bins']:,} eligible pairs. The permutation null assumes "
             "exchangeability of leaf assignments within each stratum. Other aspects of topology can still "
             "covary with estimated age. " + evidence("dating"))
    control_text = (f"The smallest primary point margin was {num(actual['difference'])} against {NAMES[strongest]}. "
                    f"Its family interval is {interval(actual['difference_ci'])}, and its order interval is "
                    f"{interval(g['order'][strongest]['difference_ci'])}. A post-hoc check counting each distinct family once "
                    f"per triple, instead of multiplying the three tip multiplicities when a family repeats, gives "
                    f"{interval(g['distinct_family'][strongest]['difference_ci'])} against the same control. "
                    "These percentile intervals condition on the modal "
                    "layers. Both intervals against untrained networks include zero with order blocks. "
                    "Both the cross-fitted and modal-layer scores exceed the registration's 0.40 smallest effect of interest. "
                    "Model-seed variability and uncertainty from selecting the layer are outside these intervals. The sample contains "
                    f"{g['sample']['eligible']:,} eligible draws at {g['sample']['nodes']} nodes. Exhaustive enumeration of "
                    f"{g['census']['n_triplets']:,} qualifying triples gives {num(g['census']['ta'])}. Transporting name-head "
                    f"labels with vectors and recomputing eligibility in every permutation gives p "
                    f"{prob(g['transported_heads']['p'])}. All comparisons concern these checkpoints and this curated name panel. " +
                    evidence("geometry"))
    sensitivity = g["secondary"]["S2"].get("within_order_permutation_p")
    scale = read_json(OUT / "scale.json")
    ab = read_json(RESULTS_DIR / MAIN / "stage4/results.json")["variants"]
    binom = ab["scientific binomials (genus-split triplets)"]
    pwr = min(ex["power"], key=lambda row: abs(row["power"] - .8))
    duplicate = {}
    for label, row in scale.items():
        duplicate.setdefault(row["matrix_sha256"], []).append(label)
    dup = "; ".join(" / ".join(names) for names in duplicate.values() if len(names) > 1)
    s2diff = fine["comparisons"][CONTROLS[1]]["difference_ci"]
    probe_text = (f"The published probe's mean held-out all-pair Pearson r was {num(probe['model']['test_test_pearson'][0])}. "
                  f"A class-centroid-plus-noise input reached {num(probe['class_blob']['test_test_pearson'][0])}; a shuffled "
                  f"target reached {num(probe['shuffled_tree']['test_test_pearson'][0])}. Within held-out strata, "
                  f"the archived probe Spearman mean was {num(probe['model']['test_test_within_stratum_spearman'][0])} "
                  f"and the independently reconstructed raw-distance comparator was {num(g['raw_probe_comparison']['mean'])}. "
                  "The class-blob baseline used empirical stratum centroids constructed on the full panel. "
                  "The training folds determined the PCA transform. The historical target scale used the maximum tree "
                  "distance over all species, and the layer was selected using the full panel. Fold predictions and fitted "
                  "fold weights were not archived. The full-panel released probe has a different training set. " +
                  src("Original probe outcomes", "results/qwen35_2b/stage2/results.json") + " " + evidence("geometry"))
    return {
        "Abstract": summary,
        "Question and prior work": ("Does the distance between two English animal names in a text model reflect their "
            "evolutionary relationship within familiar animal groups? Goodfire studied bacterial genomes in Evo 2, using "
            "angular nearest-neighbour graphs and clade holdouts. Cutinha tested embeddings of mammal descriptions. "
            "Park and colleagues studied hierarchical concepts using lexical categories. The contribution here is a "
            "within-group rooted-triplet evaluation with named controls and a frozen animal topology. "
            "[Goodfire](https://www.goodfire.com/research/phylogeny-manifold) "
            "[Cutinha](https://github.com/jadencutinha/phylotext-mammals) "
            "[Park et al.](https://arxiv.org/abs/2406.01506)"),
        "Registration and chronology": ("The initial registration preceded the first result commit. The final species "
            "freeze, nuisance controls and several analysis details followed viewing exploratory Qwen plots on the earlier "
            "panel. The habitat addendum followed the discovery plots and preceded the held-out test. The causal "
            "random-subspace count increased after the first favorable rank ordering was known. Git establishes commit "
            "order, not the earliest moment a human saw data. The November 26 calculations are retrospective. " +
            src("Registration and dated deviations", "reports/preregistration.md") + " " +
            src("Retrospective analysis protocol", "reports/reviews/2026-11-26-analysis-protocol.md") +
            " [Preregistration and prediction](https://doi.org/10.1073/pnas.1708274114)"),
        "Methods": (f"The frozen panel contains {n} species and {df_summary(g)}. For each template, we subtracted "
            "the mean last-name-token vector across species, averaged over templates, and measured cosine distance. "
            "A resolved triple has one sister pair whose common ancestor is strictly below the common ancestor of all "
            "three. We sampled up to 200 resolved triples per decision node, restricted to one eligible stratum and three "
            "distinct final name words, then averaged node means. A unique correct minimum earns one point, a unique wrong "
            "minimum zero and a tied minimum one-third. The run originally divided credit among tied winning pairs. "
            "The max-T test permutes leaf assignments within strata identically across layers. Family cross-fitting selects "
            "a layer in one half and evaluates the other half, then reverses the halves. The primary analysis averages "
            "over 20 family splits. The point estimate and its fixed-layer bootstrap target therefore differ. "
            "The primary p-value uses the observed maximum and 1,000 maxima under permutation with the plus-one "
            "Monte Carlo correction. The per-template mean used the entire name panel, so "
            "cross-fitting holds out labels for layer selection while preprocessing has access to all species vectors. "
            "Paired percentile intervals use 1,000 draws and reuse the same blocks for all controls. " +
            src("Primary registration", "reports/preregistration.md") + " " +
            "[Rooted triples](https://doi.org/10.1186/s13015-020-00175-0) "
            "[Permutation invariance](https://pmc.ncbi.nlm.nih.gov/articles/PMC6405018/)"),
        "Hypothesis decisions": headline(),
        "Primary geometry and control comparisons": controls_table(g) + "\n\n" + control_text,
        "Secondary hypotheses": secondary_table(g, d, decisions) + "\n\n" +
            f"The archived within-order analysis included {broad['triplets']:,} triples, TA {num(broad['modal'])}. "
            f"Requiring a recorded family or genus that distinguishes the sister pair retains {fine['triplets']:,}, "
            f"TA {num(fine['modal'])}. Permutations confined to the same order give exploratory layer-max p "
            f"{prob(sensitivity)}. The paired interval against Pythia step 0 is {interval(s2diff)}. "
            f"Testing against random assignment and testing for an advantage over an untrained network answer different questions. "
            f"The archived wrong-choice spelling and GloVe subsets were checked against unique "
            f"wrong choices. GloVe covered {g['glove_known']} names. A word-vector baseline measures one published "
            "co-occurrence representation; it cannot exhaust lexical or ecological cues. " + evidence("geometry") +
            "\n\n" + times,
        "Interventions and behavior": causal_table(o) + "\n\n" +
            f"The expanded run compares donor prevalence with {c['random_interchanges']} random subspaces. Its historical "
            f"Holm p {prob(c['historical_holm']['prevalence'])} belongs to that substitute endpoint. Net donor prevalence "
            f"rose by {c['net']}/{c['pairs']}. Aligned pre/post predictions for the probe and random swaps were not "
            "retained in the original result file or intervention transcript. Aggregate comparisons cannot recover the "
            "paired event or its random-subspace p-value. C-abl uses candidate-normalized sequence log-probabilities "
            f"for class words. Its raw rank p is {prob(c['ablation_rank'])}; treating the missing registered p as one "
            f"gives an upper bound of {prob(c['registered_holm']['ablation'])} on its Holm p. The rank comparison "
            f"depends on the random-subspace generator. The original transcript gives a mean captured variance "
            f"of {num(captured['random_mean'])} for random spaces, against {num(captured['target'])} for the probe, "
            f"in a span of {captured['pc_span']} PCs. Only the mean was matched. A class-only control also affected "
            "this coarse-class readout. No experiment here isolates a mechanism for within-group ancestry. " + evidence("outcomes") +
            " [Interchange interventions](https://arxiv.org/abs/2106.02997)\n\n" +
            f"The model selected the tree-closer answer on {num(o['behavior']['accuracy'])} of {o['behavior']['n']} "
            f"within-group two-choice questions, using {o['behavior']['distinct_anchors']} distinct anchors. The transcript "
            f"has {o['behavior']['ties_at_log_precision']} ties at its printed precision. The ordinary binomial p-value assumes independent "
            "trials, so we report the observed fraction descriptively. " + src("Behavioral observations", "results/qwen35_2b/stage3/results.json"),
        "Conflict stimuli": conflict_table(o) + "\n\n" +
            f"The stimulus set has {g['frozen']['conflict']['n_rows']} names and {g['frozen']['conflict']['unique_ott']} "
            f"distinct tree IDs. It shares {g['frozen']['conflict']['ott_overlap_primary']} IDs with the primary panel. "
            "Each tree choice was recalculated using ancestor depth. Spelling ties previously appeared as choices of "
            "the other partner. This hand-picked set reuses anchors and alternative species. Counts describe the items "
            "tested; they do not establish population prevalence or equal performance of the networks. " + evidence("outcomes"),
        "Habitat on held-out species": (habitat_table(o) + "\n\n" +
            f"All {e['n_heldout']} held-out species have distinct IDs from the discovery set. The panels share "
            f"{g['frozen']['heldout_eco']['family_overlap_primary']} families and "
            f"{g['frozen']['heldout_eco']['order_overlap_primary']} orders. We recovered oriented PC1 "
            f"from the discovery vectors and independently reproduced the archived test scores within "
            f"{e['axis_max_abs_error']:.2g} axis units. Marine mammals span {len(e['marine_orders'])} orders. "
            "The sensitivity test permutes scores within taxonomic orders and retains four Holm slots. Evolutionary "
            "dependence can extend within an order. EltonTraits measures foraging stratum, and aquatic words in common "
            "names can also contribute. An absent aerial-mammal group supplies no test. " + evidence("outcomes") +
            " [EltonTraits](https://esapubs.org/archive/ecol/E095/178/) [Comparative methods](https://doi.org/10.1086/284325)"),
        "Exploratory geometry, probes and power": (probe_text + "\n\n" +
            f"On {ex['pca_readings']['tree-consistent (checks 1-3 pass)']} of {ex['n_layers_pca']} layers the three "
            "numeric H1a checks pass. The contemporaneous visual description did not identify clear nested arms within "
            "groups. The full visual criterion for a nested tree therefore remains unmet. A reference overlay is fitted to the "
            f"points. The sampled-base-point relative hyperbolicity was {num(ex['gromov']['model'])} for the model "
            f"and {num(ex['gromov']['class_blob_mean'])} for class blobs. The unrooted normalized RF was "
            f"{num(ex['nj']['model'])}. Counting splits on the frozen tree gives {graph['reference_splits']} resolved "
            f"reference splits. The archived symmetric difference implies {graph['implied_shared_splits']} shared "
            f"splits with the binary neighbor-joining tree. " + evidence("graph") + " " +
            f"The single-layer simulation estimated power from {ex['power_reps']} draws per setting and tested one "
            f"permutation criterion. Estimated detection probability {num(pwr['power'])} at simulated mean TA "
            f"{num(pwr['ta_within_mean'])} has exact 95% interval {interval(pwr['binomial_ci'])}. "
            "Power for the joint H1 criteria and S5 remains unknown. " + evidence("outcomes") +
            " [Hierarchical probes](https://aclanthology.org/D19-1275/)"),
        "Ablations, sensitivity and model scale": (
            f"At the common-name H1 layer, replacing common names with scientific binomials on the same "
            f"{binom['n_head_disjoint']:,} eligible triples changes accuracy from "
            f"{num(binom['head_disjoint_common_names_modal'])} to {num(binom['head_disjoint_binomial_modal'])}. "
            f"Uncentered cosine gives {num(ab['raw cosine (no centering)']['ta_within_modal'])} and the name-token mean "
            f"{num(ab['mean over name tokens']['ta_within_modal'])} on the main subset. These are fixed-layer, "
            "descriptive contrasts. Perturbation draws varied triplet samples, prompt subsets and species. The sets "
            "of controls examined varied by perturbation. These draws do not estimate between-training-seed variation. " +
            src("Ablations", "results/qwen35_2b/stage4/results.json") + " " + src("Sensitivity", "results/qwen35_2b/stage6/results.json") +
            f"\n\nAll {len(scale)} archived scale rows were recalculated from ordered name vectors. "
            f"Identical float32 distance-array digests occur for {dup}. These rows duplicate measurements. Pythia "
            "checkpoint points share a training trajectory, and final weights in several configs are referenced by "
            "release aliases. The study cannot isolate a causal effect of model size. " + evidence("scale") +
            " [Pythia training suite](https://arxiv.org/abs/2304.01373)"),
        "Appendix and reproducibility": (
            f"At the embedding layer, {graph0['zero_pairs']} distinct name pairs have exactly zero angular distance. "
            f"Preserving those sparse edges and excluding self by row index gives minimum connected k={graph0['k']} "
            f"(archived {graph0['archived_k']}) and graph-distance Spearman {num(graph0['geodesic_rho'])} "
            f"(archived {num(graph0['archived_geodesic_rho'])}). At the H1 layer k={graph_l['k']}, "
            f"graph Spearman is {num(graph_l['geodesic_rho'])} and angular Spearman is {num(graph_l['angular_rho'])}. "
            + evidence("graph") + "\n\n" +
            "The public archive includes the frozen names, tree topology, activations, unaltered historical summaries, "
            "the independent numerical replay, and the post-hoc protocol. The TimeTree-derived individual node ages "
            "are local and must be refetched for a raw replay. The printed behavioral log has lower precision than "
            "the model scores. Historical fold-level probe predictions and the aligned intervention trials are missing. "
            "Dataset hashes authenticate file identity. They cannot establish that the models had never encountered "
            "these species, their names or their relationships in pretraining. The study has no independent held-out "
            "language, unseen clade replication of H1, or population sample of training seeds.\n\n"
            "The command `uv run --frozen python -m phylogeo.verification --render-only` rebuilds the public output "
            "from saved statistics. Omitting the flag replays statistics from archived inputs. " +
            src("Reproduction instructions", "docs/pipeline.md") + " " + src("Claim and source register", "reports/reviews/2026-11-26-claim-sources.md") +
            " [Selection bias](https://doi.org/10.1038/nn.2303) [Exchangeability](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)"),
        "Limitations and conclusion": ("Within-group agreement is an association with a sampled synthetic topology. "
            "Edge counts depend on taxon sampling. A rank or a depth in this tree is not elapsed evolutionary time. "
            "Selection on English common names and Wikipedia sitelinks, repeated clades, and shared ecological and lexical "
            "traits limit generalization. Untrained comparisons refer to individual checkpoints. A confidence interval "
            "containing chance or zero leaves a range of plausible effects; equivalence would require an explicit margin. "
            "Neither the distance correlations nor the intervention archive identify a phylogeny-specific causal mechanism. "
            "[Open Tree](https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3) "
            "[TimeTree](https://doi.org/10.1093/molbev/msac174) [Equivalence tests](https://doi.org/10.1177/1948550617697177)")}


def df_summary(g):
    return f"{g['sample']['families']} families in {g['sample']['orders']} orders"


def write_review():
    g, d, o, decision = assemble()
    save("decisions", decision)
    sections = section_texts()
    text = "# Independent scientific audit, 26 November 2026\n\nLibor “Burny” Burian.\n\n"
    text += "\n\n".join(f"## {name}\n\n{value}" for name, value in sections.items())
    old = read_json(RESULTS_DIR / MAIN / "stage1b/results.json")
    eco = read_json(RESULTS_DIR / MAIN / "eco/results.json")["tests"]
    new = decision["secondary_holm"]
    change = [
        ["H1, cross-fitted TA", num(old["H1"]["crossfit_ta_within"]), num(g["models"][MAIN]["crossfit"]), "Written tie rule"],
        ["S2, eligible triples", old["secondary"]["S2"]["n_triplets"], g["secondary"]["S2"]["triplets"], "Family/genus split required"],
        ["S3, eligible triples", old["secondary"]["S3"]["n_triplets"], g["secondary"]["S3"]["triplets"], "Strictly wrong closest pair"],
        ["S5, before Holm", prob(old["secondary"]["S5"]["maxT_p"]), prob(d["partial"]["p_global_max"]), "Fixed layer replaced by an all-layer test"],
        ["C-swap, Holm p", prob(o["causal"]["expanded"]["historical_holm"]["prevalence"]), "Unevaluable", "New switches require aligned predictions"],
        ["M1, Holm p", prob(eco["M1_marine_lt_ground"]["holm_p"]), prob(o["habitat"]["holm"]["M1"]), "Four registered slots"],
        ["B1, Holm p", prob(eco["B1_water_negative"]["holm_p"]), prob(o["habitat"]["holm"]["B1"]), "Four registered slots"],
    ]
    change.extend([[k + ", Holm p", prob(old["secondary"][k]["holm_p"]), prob(new[k]), "Recomputed five-test family"]
                   for k in ("S1", "S2", "S3", "S4", "S5")])
    text += "\n\n## Original and corrected measurements\n\n" + table(
        ["Quantity", "Originally reported", "Independent reanalysis", "Reason"], change)
    text += "\n\n" + src("Original geometry", "results/qwen35_2b/stage1b/results.json") + " " + evidence("geometry")
    text += " " + evidence("dating") + " " + evidence("outcomes") + ".\n"
    (REPO_ROOT / REVIEW).write_text(text.rstrip() + "\n", encoding="utf-8")
