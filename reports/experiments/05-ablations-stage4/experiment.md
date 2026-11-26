# Experiment 05 (stage 4): ablations

Common and scientific names must be compared on the same eligible triples at the same layer. The result changes with name choice and token readout. A point estimate close to one-third does not establish equivalence to chance. The probe variants are recorded outcomes from the original fits. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#ablations-sensitivity-and-model-scale) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/05-ablations-stage4/experiment.md)

**Status:** done 2026-11-20 (exploratory).
- **Commands:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases stage4
  ```
  Extra extractions, run first:
  - `--phases extract --name-sets species_sci` (scientific binomials);
  - `configs/qwen35_2b_raw.yaml` (no chat template).
- **Numbers:** auto-generated in [`results/qwen35_2b/SUMMARY.md`](../../../results/qwen35_2b/SUMMARY.md), section "Stage 4".

## What changes, one at a time
- Readout: last name token vs mean over name tokens.
- Distance normalization: per-template centering (main) vs raw cosine, z-scored dimensions, stratum-centered.
- Prompt format: chat template vs raw text.
- Names: common names vs scientific binomials. Binomial triplets are split above genus, because congeners share the genus token.
- Species set: polysemous names included (414 species, combined Open Tree tree).
- Probe:
  - dimension k ∈ {2, 3, 5, 10, 20};
  - metric angular / Euclidean / Poincaré at k ∈ {2, 5, 10};
  - within-group held-out evaluation, compared with the shuffled-tree probe.
- Geodesic: the kNN k used for the graph.
