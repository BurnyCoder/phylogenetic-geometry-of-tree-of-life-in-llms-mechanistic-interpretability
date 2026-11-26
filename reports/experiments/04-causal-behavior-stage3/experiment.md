# Experiment 04 (stage 3): behaviour and causal tests

The historical swap rate counts post-edit donor predictions. The stored marginals bound newly acquired donor answers, but the per-pair random and probe changes are unavailable. The literal C-swap test is unevaluable. C-abl does not clear its random-subspace comparison. The class-only results are descriptive and the pairwise questions reuse species. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#interventions-and-behavior) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/04-causal-behavior-stage3/experiment.md)

**Status:** done 2026-11-20. This is planned run R3 in [planned-runs](../../planned-runs.md).
- **Behaviour:** exploratory.
- **C-abl and C-swap:** confirmatory, a separate Holm family of 2.
- **Class-only control:** exploratory, added by deviation 20 before the final run.

**Commands:**
```powershell
uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases stage3
```
```powershell
uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases stage3_causal
```
- `stage3` runs the behavioural tests and the first causal run. The first run is kept as `C_abl_v1` and `C_swap_v1`.
- `stage3_causal` reruns only the causal tests with the corrected design (deviations 19–20).

**Numbers:** auto-generated in [`results/RESULTS.md`](../../../results/RESULTS.md), section "Stage 3 — behavior and causal tests", from `results/qwen35_2b/stage3/results.json`. BH q-values for the exploratory tests are in `results/qwen35_2b/exploratory_bh.json`. No number in this file is typed by hand except design constants.

## Why
- **Correlation is not use.** Stages 1b and 2 show that the geometry agrees with the tree and that a 10-d flat probe predicts tree distance. Neither shows that the model *uses* that subspace.
- **The Goodfire analogue.** The Evo 2 post ([Pearce et al. 2025](https://www.goodfire.com/research/phylogeny-manifold)) reports no causal test. Manifold steering ([Wurgaft et al.](https://arxiv.org/abs/2605.05115)) and interchange interventions ([Geiger et al. 2021](https://arxiv.org/abs/2106.02997)) are the standard ways to test whether a representation is used.
- **Behaviour.** Does the model *say* which animals are related, and does that agree with its representation?

## Method
**Behaviour (exploratory)**
- *Relatedness questions.* The model is asked which of two animals is more closely related to a third, with answers A/B. Each answer letter is scored by log-probability, and both answer orders are averaged to cancel position bias.
- *Triplet sets.* 300 within-stratum species triplets (sampled as in stage 1b) and the 36 preregistered conflict triplets.
- *Class word.* Each species is asked for its class word ("What kind of animal is the {name}? Choose one of: …"). The answer is the argmax over the candidate words.
- The preserved behavioral transcript supplies rounded scores. The intervention transcript supplies aggregates.

**Causal tests (confirmatory, final design from deviations 19–20)**
- *Layer.* Layer 6, the cross-fitted modal layer from stage 1b.
- *Subspace.* A k = 10 flat probe refit on the species of one half of the families. Interventions are evaluated on the other half: 188 held-out species.
- *Readout.* The class-word question above, scored as a softmax over the candidate class words. Each word is scored by its full-sequence log-probability.
- *Intervention.* A forward hook replaces the residual stream after decoder layer 6 (`causal.py: residual_edit`).
- *C-abl.*
  - Mean-ablate the probe subspace at every name token: h' = h − UUᵀ(h − μ), where μ is the training species' mean activation at the name's last token in the class prompt. Measure the drop in log P(correct class).
  - Compare with 99 random 10-d subspaces drawn inside the span of the top-m principal components of the training species. The value m = 64 is chosen so that their captured variance matches the probe's on average.
  - Primary statistic: the rank p-value (1 + #random subspaces with drop ≥ probe) / 100. Secondary: a one-sided Wilcoxon test over species, probe drop vs mean random drop.
- *C-swap.*
  - 200 pairs (A, B) of held-out species with different class words. At B's last name token, replace the subspace component with A's (taken from A's last name token in the same prompt).
  - The registered endpoint counts new switches to A's class. The archived implementation counted every post-edit answer in A’s class. Compare with the same 99 random subspaces using the rank p-value.
- *Correction.* The original Holm family used C-abl and post-edit donor prevalence.
- *Off-target check.* Next-token loss on 20 neutral sentences, token-weighted. It is computed by one function for no edit, probe ablation and random ablation (the mean of the first 10 random subspaces), with the ablation applied at every token.
- *Class-only control (deviation 20).*
  - The same k = 10 probe, trained on the class-oracle distance (0 within a stratum, 1 between), with the same training species, steps and seed.
  - Rule: a phylogeny-specific causal role is claimed only if the phylogeny probe beats it on both tests at p < 0.05. Ablation is tested with a one-sided Wilcoxon over species, and interchange with a one-sided exact McNemar over the same pairs.

**Planned but not run:** R3-steer (geodesic vs linear-chord steering) and R3-sae (SAE features; no SAE exists for post-trained Qwen3.5-2B). Both are in the README to-do.
