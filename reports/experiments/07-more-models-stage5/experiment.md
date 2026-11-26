# Experiment 07 (stage 5): more models (scale, architecture, training dynamics)

The audit replayed each row from saved vectors. The step-zero and step-one Pythia-1.4B distance matrices match exactly, as do the final and last checkpoint rows. The Pythia series describes one training trajectory. Between-model comparisons cannot isolate the effect of size or architecture. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#ablations-sensitivity-and-model-scale) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/07-more-models-stage5/experiment.md)

**Status:** done 2026-11-20 (descriptive; no confirmatory test). This is planned run R5 in [planned-runs](../../planned-runs.md).

**Commands:** for each config the chain runs `extract`, then `scale`, then the summary:
```powershell
uv run python -m phylogeo.run --config configs/<run>.yaml --phases extract,scale
```
```powershell
uv run python -m phylogeo.run --phases scale_summary
```

**Numbers:** auto-generated in [`results/RESULTS.md`](../../../results/RESULTS.md), section "Stage 5", from `results/stage5/summary.json`. Figures are in `results/stage5/`.

## Why
- **Scale within a family.** Does within-group tree order grow with model size? This uses Qwen3.5-0.8B vs 2B, and Pythia from 70M to 2.8B. The Pythia checkpoints share the Pile training setting.
- **Architecture.** Is the effect specific to Qwen3.5's hybrid Gated-DeltaNet attention? Qwen3-1.7B is a standard full-attention model of similar size.
- **Training dynamics.** When does the structure appear during training? This uses 12 log-spaced Pythia-1.4B checkpoints, from step 0 (untrained) to 143,000.

## Method
- **Same setup for every model.** Identical species, templates, triplets (the stage-1b sample, same seed) and metric.
- **Prompt format.** Chat format for the post-trained Qwen models; raw text with BOS for Pythia.
- **Statistics per run.**
  - TA_within at every layer.
  - A max-T permutation p-value with 199 within-stratum permutations. The floor is 0.005; a lighter null than stage 1b, because these runs are descriptive.
  - The family cross-fitted TA_within with 10 splits.
  - The Mantel correlation at the cross-fitted layer.
- **Comparison lines.** The untrained references (random-init Qwen3.5-2B, Pythia-410m step 0) and the lexical baselines from stage 1b are drawn as reference lines. Pythia-1.4B step 0 is the untrained reference for the checkpoint series.
