---
license: apache-2.0
tags: [interpretability, activations, phylogeny, representation-geometry]
---
# Hidden-state readouts for animal species names

The files contain saved floating-point readouts. Layout is `<run>/<name_set>/last.npy`, `mean.npy` and `names.txt` where available. Rows follow the accompanying name file. Array dimensions are names × prompts × hidden-state indices × hidden width. Index zero is the embedding output and the final hidden state can include the model's final normalization. `last` selects the last name token and `mean` averages over the name span. [Extraction implementation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/src/phylogeo/extract.py)

Run configurations identify the model, revision field and prompt format. The main Qwen revision is an immutable SHA. Several Pythia settings use release aliases. The archive includes trained and random controls and a sequence of Pythia training checkpoints. Identical distance arrays for some entries are documented in the independent replay. [Configurations](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/configs) [Frozen-data calculation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/scale.json)

Name sets include the primary panel, scientific names, flagged polysemous names, conflict items, held-out habitat species and calendar examples, depending on run. The conflict items share tree IDs with the primary panel. The held-out habitat set has disjoint tree IDs. Published tests concern name representations and the sampled reference tree. [Frozen panel manifest](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data/FREEZE.json) [Independent scientific audit](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-scientific-audit.md)
