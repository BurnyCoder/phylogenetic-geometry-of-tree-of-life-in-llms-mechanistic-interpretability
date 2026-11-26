"""Generate release documentation from archived data without loading checkpoint weights."""

from phylogeo.config import REPO_ROOT, RESULTS_DIR
from .publication import REVIEW, evidence, num, src
from .statistics import MAIN, OUT, read_json


def model_card(conf):
    p = read_json(OUT / "outcomes.json")["exploratory"]["probe"]
    raw = read_json(OUT / "geometry.json")["raw_probe_comparison"]["mean"]
    sensitivity = read_json(RESULTS_DIR / MAIN / "stage6/results.json")["probe_seed"]["test_test_pearson"]
    return f"""---
license: apache-2.0
base_model: {conf['model_id']}
tags: [interpretability, probing, phylogeny, representation-geometry]
---
# Flat-subspace probe for animal names in {conf['model_id']}

This checkpoint maps the last name-token readout from layer {conf['layer']} to {conf['k']} coordinates. Its target is edge-count path length on the sampled Open Tree topology. Its training-set Pearson correlation is {num(conf['fit_pearson_all_species'])} across the {conf['n_species']} fit species. [Checkpoint configuration](https://huggingface.co/BurnyCoder/phylogeo-flat-probe-qwen3.5-2b/blob/main/config.json) [Method inspiration](https://www.goodfire.com/research/phylogeny-manifold)

For each template, center the readouts over the reference species and then average templates. The stored parameters implement `x_pca = (x - pca_mean) @ pca_components.T / z_scale` and `z = (x_pca - b) @ W.T`. Convert angular separation to estimated tree units with `exp(log_beta) * arccos(clip(cos(z_i, z_j), -1 + 1e-6, 1 - 1e-6)) / target_scale`. The archive contains the weights and PCA in `probe.safetensors` and in-sample coordinates in `species_coordinates.csv`. {src('Probe implementation', 'src/phylogeo/probes/flat.py')}

The release script fit the probe with full-batch Adam on all unordered training pairs, using a squared distance error plus a reconstruction penalty. It used 2,000 optimization steps and seed 0. The checkpoint configuration records the base revision, readout, dimension and scale factors. {src('Release recipe', 'src/phylogeo/release.py')} {src('Objective and optimizer', 'src/phylogeo/probes/flat.py')}

Separate historical cross-validation runs yielded mean all-pair Pearson r {num(p['model']['test_test_pearson'][0])} for held-out families, {num(p['class_blob']['test_test_pearson'][0])} for a class-centroid-plus-noise input, and {num(p['shuffled_tree']['test_test_pearson'][0])} for shuffled targets. Mean within-stratum held-out Spearman was {num(p['model']['test_test_within_stratum_spearman'][0])} for the probe and {num(raw)} for raw distances on the same pairs. The historical procedure chose the layer on the full panel and used the full-panel maximum for target scaling. The released checkpoint used all species. Each cross-validation run had its own fits. {evidence('geometry')} {src('Archived probe outcomes', 'results/qwen35_2b/stage2/results.json')}

In a separate archived sweep, changing the family allocation and probe initialization together gave mean held-out all-pair Pearson r {num(sensitivity['mean'])} with SD {num(sensitivity['sd'])} across {len(sensitivity['values'])} settings. Replacing English common names with scientific binomials also reduced the fixed-layer triplet score on matched triples. {src('Seed sweep', 'results/qwen35_2b/stage6/results.json')} {src('Name ablations', 'results/qwen35_2b/stage4/results.json')}

The intervention archive cannot identify the registered comparison of newly acquired donor answers against random subspaces. No model or probe was evaluated afresh in the November 26 reanalysis. {src('Independent scientific audit', REVIEW)}
"""


def activation_card():
    return f"""---
license: apache-2.0
tags: [interpretability, activations, phylogeny, representation-geometry]
---
# Hidden-state readouts for animal species names

The files contain saved float32 readouts. Layout is `<run>/<name_set>/last.npy`, `mean.npy` and `names.txt` where available. Rows follow the accompanying name file. Array dimensions are names × prompts × hidden-state indices × hidden width. Index zero is the embedding output and the final hidden state can include the model's final normalization. `last` selects the last name token and `mean` averages over the name span. {src('Extraction implementation', 'src/phylogeo/extract.py')}

Run configurations identify the model, revision field and prompt format. The main Qwen revision is an immutable SHA. Several Pythia settings use release aliases. The archive includes trained and random controls and a sequence of Pythia training checkpoints. Identical distance arrays for some entries are documented in the independent replay. {src('Configurations', 'configs')} {evidence('scale')}

For example, the primary arrays are under `qwen35_2b/species/` and its random-weight comparison under `qwen35_2b_random/species/`. Other run directories distinguish raw-text prompts (`qwen35_2b_raw`), final and step-zero Pythia controls, and training checkpoints (`pythia14b_step*`). The exact name-set keys include `species_sci`, `species_polysemous`, `conflict`, `heldout_eco`, `days` and `months` where present. Read a slice with `numpy.load(path, mmap_mode="r")` and get the row labels from `names.txt` in the same directory. Wikidata supplies the source names. [Published file tree](https://huggingface.co/datasets/BurnyCoder/phylogenetic-geometry-activations/tree/main) [Wikidata terms](https://www.wikidata.org/wiki/Wikidata:Licensing) {src('Source data', 'data')}

Name sets include the primary panel, scientific names, flagged polysemous names, conflict items, held-out habitat species and calendar examples, depending on run. The conflict items share tree IDs with the primary panel. The held-out habitat set has disjoint tree IDs. Published tests concern name representations and the sampled reference tree. {src('Frozen panel manifest', 'data/FREEZE.json')} {src('Independent scientific audit', REVIEW)}
"""


def species_card():
    g = read_json(OUT / "geometry.json")["frozen"]
    return f"""---
license: cc0-1.0
pretty_name: Phylogenetic geometry species set ({g['species']['n_rows']} animals + Open Tree topology)
tags: [interpretability, phylogenetics, tree-of-life, llm]
size_categories: [n<1K]
---
# Frozen animal names and reference topologies

The primary panel contains {g['species']['n_rows']} English common names with {g['species']['unique_ott']} distinct Open Tree IDs. Selection used Wikidata entries, English Wikipedia titles, sitelink counts, taxonomic strata and a family cap. Ranks use NCBI first with an Open Tree fallback. This is a curated sample of English names. The supplied topology describes the induced synthetic Open Tree on the selected taxa. {src('Selection rules and amendments', 'reports/preregistration.md')} {src('Species table', 'data/species.csv')} [Open Tree](https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3)

| File in this release | Contents |
|---|---|
| `species.csv`, `species_tree.nwk` | Primary English names and the induced Open Tree topology. Newick tip labels use `ott` followed by the integer in `ott_id`. |
| `species_polysemous.csv`, `species_polysemous_tree.nwk` | Flagged names retained by the sensitivity selection and their topology. |
| `conflict_triplets.csv` | Frozen question ID, anchor, proposed phylogenetic and folk partners, scientific names and conflict category. |
| `conflict.csv`, `conflict_tree.nwk` | The mapping from challenge names to taxa and the topology used to resolve each question. |
| `FREEZE.json` | Row counts and SHA-256 hashes for the CSV and Newick pairs listed in this release's manifest. |

In the primary CSV, `common_name` and `sci_name` give the English and scientific names. `wikidata`, `ncbi` and `ott_id` identify the taxon in each source. `sitelinks` counts Wikimedia sitelinks on the Wikidata item. `stratum` is the project's sampling group. Columns from `phylum` through `genus` are the ranks used in the analysis, and their `ott_` counterparts preserve the Open Tree values. `n_words` counts whitespace-separated words. `head_noun` is the lowercased final word or hyphen component. `polysemous` records the study's ambiguity rule. {src('Panel builder and column definitions', 'src/phylogeo/data/species.py')}

The conflict file has {g['conflict']['n_rows']} name rows, {g['conflict']['unique_ott']} distinct IDs and {g['conflict']['ott_overlap_primary']} IDs in common with the primary panel. The topology determines which alternative is closer in a resolved item. The separate polysemy sensitivity set contains {g['species_polysemous']['n_rows']} names. The project's held-out ecology panel has {g['heldout_eco']['n_rows']} species and {g['heldout_eco']['ott_overlap_primary']} IDs in common with the primary set. Consult the project data directory for the held-out files and the current manifest. The hashes cover the listed CSV/Newick pairs. {src('Data directory', 'data')} {src('Freeze manifest', 'data/FREEZE.json')} {evidence('geometry')}

Libor “Burny” Burian curated the release. Wikidata makes its structured data available under CC0. Open Tree's CC0 dedication is subject to pre-existing data terms. [Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing) [Open Tree data rights](https://tree.opentreeoflife.org/opentree/about/licenses) [Open Tree synthesis and citation](https://tree.opentreeoflife.org/about/open-tree-of-life)

The project keeps individual TimeTree ages in a separate local cache. Edge-count distances and estimated divergence times answer different questions. EltonTraits supplies the held-out foraging measurements under the source terms listed by the publisher. [TimeTree 5](https://doi.org/10.1093/molbev/msac174) [EltonTraits](https://esapubs.org/archive/ecol/E095/178/) {src('Independent scientific audit', REVIEW)}
"""


def write_cards():
    directory = REPO_ROOT / "reports/cards"
    directory.mkdir(exist_ok=True)
    for name, value in (("probe", model_card(read_json(OUT / "release-config.json"))),
                        ("species", species_card()), ("activations", activation_card())):
        (directory / (name + ".md")).write_text(value, encoding="utf-8")
