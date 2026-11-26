# Experiment 08 (stage 6): sensitivity

The recorded draws vary sampling, templates and probe seeds. Control coverage varies across perturbations. These summaries give within-protocol stability. They do not measure variation across independently trained base models. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#ablations-sensitivity-and-model-scale) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/08-sensitivity-stage6/experiment.md)

**Status:** done 2026-11-20. The run is planned run R6-sens in [planned-runs](../../planned-runs.md).
- **Command:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases stage6
  ```
  It runs on the CPU, using the stored activations.
- **Numbers:** auto-generated in [`results/qwen35_2b/SUMMARY.md`](../../../results/qwen35_2b/SUMMARY.md), section "Stage 6".
- **Figure:** [`sensitivity_ta_within.png`](../../../results/qwen35_2b/stage6/sensitivity_ta_within.png).

## Why
Every headline number in stages 1b and 2 came from one fixed draw of arbitrary choices:
- the triplet sample;
- the 8 templates;
- the 400 species;
- the family split used for cross-fitting;
- the probe seed.

A result that depends on one of these draws is not a result. Each source is varied **on its own**, with everything else held at the main-run values.

## What was varied
| Source | Repetitions | What is recomputed |
|---|---|---|
| Triplet sampling seed | 20 | TA_within for the model and every control on the same draw |
| Triplets per decision node K | 50, 100, 200, 400 | Same |
| Templates: random halves (4 of 8) | 20 | Model TA_within, best layer, Mantel |
| Templates: leave one out | 8 | Same |
| Templates: single template | 8 | Same |
| Species: random 80% subsamples | 20 | Tree re-induced and triplets re-sampled; model and controls on the same draw; Mantel |
| Cross-fitting family-split seed | 20 (each with 20 splits) | Cross-fitted TA_within and modal layer |
| Flat-probe seed (sets both folds and initialization) | 20 fresh seeds, disjoint from the stage-2 seeds | Held-out-family r; within-stratum held-out Spearman |

The controls recomputed on every triplet and species draw are random-init Qwen3.5-2B, Pythia-410m step 0, character 3-grams, name token count and log popularity. The "margin" is the model's TA_within minus the **best** of these controls on the same draw.
