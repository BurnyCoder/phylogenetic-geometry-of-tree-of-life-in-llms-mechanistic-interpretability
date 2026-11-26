# Experiment 02 (stage 1b): triplet accuracy, Mantel, controls, conflict triplets

The primary decision survives the written tie rule with family blocks. The margin over each untrained network becomes uncertain with order blocks. The narrow family/genus interpretation of S2 meets its chance-null criterion. S3, S4 and adjusted S5 do not. Spelling ties on conflict stimuli are reported separately. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#primary-geometry-and-control-comparisons) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/02-triplets-stage1b/experiment.md)

**Status:** done 2026-11-20 on the re-frozen species set.
- **Command:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases analyze,conflict
  ```
- **Numbers:** all in the auto-generated [`results/qwen35_2b/SUMMARY.md`](../../../results/qwen35_2b/SUMMARY.md) (from `results/qwen35_2b/stage1b/results.json` and `conflict/conflict.json`). This page interprets them and types no numbers by hand.

## What was tested (preregistered)
- **Representation.** Qwen3.5-2B residual stream at the last name token. Prompts are in the chat template, 8 templates per species, mean-centred per template, compared by cosine distance, at all 25 layers.
- **Primary metric (H1).** Node-balanced rooted-triplet accuracy over triplets whose three species share a stratum and share no head noun (TA_within).
  - Significance: max-T permutation over layers.
  - Point estimate: family-blocked cross-fitting.
  - CI: clade-block bootstrap.
- **Required paired comparisons** (preregistered, plus deviations 11–12):
  - random-init Qwen3.5-2B;
  - Pythia-410m step0;
  - character 3-grams;
  - the input embedding (layer 0);
  - name token count;
  - log Wikipedia popularity.
- **Secondary hypotheses** (Holm across S1–S5):
  - S1: global Mantel correlation with tree path length;
  - S2: distinct final name words within one order, with the sister family or genus distinct from the outgroup under the narrow reading;
  - S3: lexical-discordant triplets;
  - S4: co-occurrence (GloVe)-discordant triplets;
  - S5: within-rank association with divergence time after adjustment for MRCA depth.

## Where the signal appears

The family-block margin is smallest against untrained Pythia. Its own layer-max test gives evidence of within-group ordering even before pretraining. Random Qwen also has a selected-layer estimate above one-third, with a larger layer-max p-value. Per-stratum estimates are clearest for birds and mammals. Groups with few eligible nodes have correspondingly uncertain estimates. The full breakdown reports node counts, sampling intervals, and the rank and tree oracles. [Generated results](../../../results/qwen35_2b/SUMMARY.md#stage-1b)

The spelling-neutral and spelling-discordant analyses ask different questions. Removing every shared character trigram leaves a positive trained-minus-untrained margin. Requiring spelling to prefer a specific wrong pair yields a different subset, and its adjusted test falls short. The hand-selected conflict examples also reuse species and sometimes give tied spelling distances. Frequencies of choices there cannot identify a learned folk taxonomy. [Subset replay](../../../results/scientific-verification/geometry.json) · [Item-level conflict choices](../../../results/scientific-verification/outcomes.json)
