# Experiment 08 (stage 6): sensitivity

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

## Results in words
- **Triplet sampling and K barely matter.** The spread across triplet seeds is tiny compared with the gap to the best control, and K changes nothing material.
- **Templates barely matter.** Any half of the templates, any leave-one-out set and even a **single** template give nearly the same TA_within. The selected layer is the same (layer 6) for almost every template subset. A few single templates peak in a late layer instead.
- **Species subsampling is the largest source of spread, but the model wins every time.** Removing a random 20% of species moves TA_within the most of all perturbations. On every one of the 20 draws the model still beats the best control on the same draw.
- **The cross-fitted estimate and its layer are stable** across family-split seeds; the modal layer is 6 in almost all of them.
- **The probe is stable across seeds,** both overall and within strata.

## Interpretation
The primary result, within-stratum tree order beyond untrained networks and name statistics, is not an artifact of one triplet sample, one prompt set, one species list or one split. The main uncertainty comes from *which species* are in the set. This is what the family-block bootstrap CI of stage 1b already models.

## Learnings
- Recompute the controls on the *same* perturbed draw and report the paired margin. The marginal spreads alone overstate uncertainty about the comparison.
- With per-template centering and a last-token readout, the prompt wording is a minor factor for this model, so the representation is largely prompt-invariant.
