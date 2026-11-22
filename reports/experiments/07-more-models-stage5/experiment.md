# Experiment 07 (stage 5): more models (scale, architecture, training dynamics)

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
- **Scale within a family.** Does within-group tree order grow with model size? This uses Qwen3.5-0.8B vs 2B, and Pythia from 70M to 2.8B, all trained on one corpus, the Pile.
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

## Results in words
Numbers are in [`results/RESULTS.md`](../../../results/RESULTS.md), section "Stage 5". Figures: [`scale.png`](../../../results/stage5/scale.png) and [`training_dynamics.png`](../../../results/stage5/training_dynamics.png).

1. **Scale (Pythia, one corpus).**
   - Within-group triplet accuracy rises with size from 70M to 2.8B, roughly monotonically. 160M sits below 70M, within noise.
   - The two smallest Pythia models stay below the untrained Pythia-410m step-0 control from stage 1b. Only from about 410M do trained Pythia models clearly exceed the untrained references.
   - Pythia-2.8B reaches about the level of Qwen3.5-2B.
2. **Architecture.**
   - Qwen3-1.7B (standard full attention) is close to Qwen3.5-2B (hybrid Gated-DeltaNet).
   - Qwen3.5-0.8B is lower.
   - The effect is therefore not specific to the hybrid architecture, and within the Qwen3.5 family it grows with size.
3. **Training dynamics (Pythia-1.4B).**
   - At initialization and over the first 512 steps, accuracy is at chance and the max-T test is not significant.
   - It jumps by step 1000 and then keeps rising slowly to the final checkpoint.
   - The Mantel correlation with path length rises over the same period, from about zero to its final value.
   - Within-group tree order is therefore *learned* from text and keeps improving late in training. An untrained network does not provide it.
4. **Untrained references differ by architecture.** Pythia-1.4B at step 0 is near chance. Untrained Pythia-410m is above chance by its own max-T test. Random-init Qwen3.5-2B is not, once the layer choice is accounted for (corrected by the 2026-11-22 audit; its value at its selected layer is above 1/3). The required untrained controls therefore matter, and a single "untrained baseline" number should not be generalized across architectures.

## Caveats
- **Lighter analysis.** Only 199 permutations (p floor 0.005), cross-fitting with 10 splits and no bootstrap CIs. For the main model this gives a slightly different cross-fitted value than stage 1b.
- **Untrained twins.** Only Pythia-410m and Pythia-1.4B have untrained versions in this study.
- **Layer counts.** These differ across models, so the best layer is compared by relative depth.

## Audit note (2026-11-22)
From the [independent audit](../../reviews/2026-11-22-independent-audit.md):
- **Reference lines are now like-for-like.** The figure plots cross-fitted values for every model, but the untrained reference lines used each control's full-sample value at its selected layer, which is optimistic. They are now cross-fitted too. Point 1's comparison of the two smallest Pythia models with the untrained Pythia-410m therefore reads "about equal" rather than "below".
- **One size accounting.** Qwen parameter counts were checkpoint totals including a vision encoder and an MTP head, while Pythia used nominal sizes. All models now use nominal sizes from the model names.
- **Steps 0 and 1 of Pythia-1.4B give identical numbers** because their stored activations are identical. The two checkpoint files differ on the Hub. Plausibly, the first update at the warm-up learning rate is far below bfloat16 resolution, and the models are loaded in bfloat16. This was not verified at the weight level.
- **Cross-fitting here uses 10 family splits**, so Qwen3.5-2B's value differs slightly from the stage-1b value with 20 splits; the key-results table now says so.
