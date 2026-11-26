---
license: cc0-1.0
pretty_name: Phylogenetic geometry species set (400 animals + Open Tree topology)
tags: [interpretability, phylogenetics, tree-of-life, llm]
size_categories: [n<1K]
---
# Frozen animal names and reference topologies

The primary panel contains 400 English common names with 400 distinct Open Tree IDs. Selection used Wikidata entries, English Wikipedia titles, sitelink counts, taxonomic strata and a family cap. Ranks use NCBI first with an Open Tree fallback. This is a curated sample of English names. The supplied topology describes the induced synthetic Open Tree on the selected taxa. [Selection rules and amendments](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/preregistration.md) [Species table](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data/species.csv) [Open Tree](https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3)

| File in this release | Contents |
|---|---|
| `species.csv`, `species_tree.nwk` | Primary English names and the induced Open Tree topology. Newick tip labels use `ott` followed by the integer in `ott_id`. |
| `species_polysemous.csv`, `species_polysemous_tree.nwk` | Flagged names retained by the sensitivity selection and their topology. |
| `conflict_triplets.csv` | Frozen question ID, anchor, proposed phylogenetic and folk partners, scientific names and conflict category. |
| `conflict.csv`, `conflict_tree.nwk` | The mapping from challenge names to taxa and the topology used to resolve each question. |
| `FREEZE.json` | Row counts and SHA-256 hashes for the CSV and Newick pairs listed in this release's manifest. |

In the primary CSV, `common_name` and `sci_name` give the English and scientific names. `wikidata`, `ncbi` and `ott_id` identify the taxon in each source. `sitelinks` counts Wikimedia sitelinks on the Wikidata item. `stratum` is the project's sampling group. Columns from `phylum` through `genus` are the ranks used in the analysis, and their `ott_` counterparts preserve the Open Tree values. `n_words` counts whitespace-separated words. `head_noun` is the lowercased final word or hyphen component. `polysemous` records the study's ambiguity rule. [Panel builder and column definitions](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/src/phylogeo/data/species.py)

The conflict file has 79 name rows, 78 distinct IDs and 45 IDs in common with the primary panel. The topology determines which alternative is closer in a resolved item. The separate polysemy sensitivity set contains 14 names. The project's held-out ecology panel has 159 species and 0 IDs in common with the primary set. Consult the project data directory for the held-out files and the current manifest. The hashes cover the listed CSV/Newick pairs. [Data directory](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data) [Freeze manifest](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data/FREEZE.json) [Frozen-data calculation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/geometry.json)

Libor “Burny” Burian curated the release. Wikidata makes its structured data available under CC0. Open Tree's CC0 dedication is subject to pre-existing data terms. [Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing) [Open Tree data rights](https://tree.opentreeoflife.org/opentree/about/licenses) [Open Tree synthesis and citation](https://tree.opentreeoflife.org/about/open-tree-of-life)

The project keeps individual TimeTree ages in a separate local cache. Edge-count distances and estimated divergence times answer different questions. EltonTraits supplies the held-out foraging measurements under the source terms listed by the publisher. [TimeTree 5](https://doi.org/10.1093/molbev/msac174) [EltonTraits](https://esapubs.org/archive/ecol/E095/178/) [Independent scientific audit](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-scientific-audit.md)
