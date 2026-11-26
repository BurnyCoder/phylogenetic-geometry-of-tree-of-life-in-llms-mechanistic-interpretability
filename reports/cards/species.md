---
license: cc0-1.0
pretty_name: Phylogenetic geometry species set (400 animals + Open Tree topology)
tags: [interpretability, phylogenetics, tree-of-life, llm]
size_categories: [n<1K]
---
# Frozen animal names and reference topologies

The primary panel contains 400 English common names with 400 distinct Open Tree IDs. Selection used Wikidata entries, English Wikipedia titles, sitelink counts, taxonomic strata and a family cap. Ranks use NCBI first with an Open Tree fallback. This is a curated sample of English names. The supplied topology describes the induced synthetic Open Tree on the selected taxa. [Selection rules and amendments](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/preregistration.md) [Species table](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data/species.csv) [Open Tree](https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3)

The conflict file has 79 name rows, 78 distinct IDs and 45 IDs in common with the primary panel. The topology determines which alternative is closer in a resolved item. The separate polysemy sensitivity set contains 14 names. The project's held-out ecology panel has 159 species and 0 IDs in common with the primary set. Consult the project data directory for the held-out files and the current manifest. The hashes cover the listed CSV/Newick pairs. [Data directory](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data) [Freeze manifest](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data/FREEZE.json) [Frozen-data calculation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/geometry.json)

The project keeps individual TimeTree ages in a separate local cache. Edge-count distances and estimated divergence times answer different questions. EltonTraits supplies the held-out foraging measurements under the source terms listed by the publisher. [TimeTree 5](https://doi.org/10.1093/molbev/msac174) [EltonTraits](https://esapubs.org/archive/ecol/E095/178/) [Independent scientific audit](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-scientific-audit.md)
