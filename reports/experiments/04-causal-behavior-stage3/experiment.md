# Experiment 04 (stage 3): behaviour and causal tests

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
- Every prompt and output is logged under `logs/`.

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
  - Count how often the answer switches to A's class. Compare with the same 99 random subspaces using the rank p-value.
- *Correction.* Holm over C-abl and C-swap.
- *Off-target check.* Next-token loss on 20 neutral sentences, token-weighted. It is computed by one function for no edit, probe ablation and random ablation (the mean of the first 10 random subspaces), with the ablation applied at every token.
- *Class-only control (deviation 20).*
  - The same k = 10 probe, trained on the class-oracle distance (0 within a stratum, 1 between), with the same training species, steps and seed.
  - Rule: a phylogeny-specific causal role is claimed only if the phylogeny probe beats it on both tests at p < 0.05. Ablation is tested with a one-sided Wilcoxon over species, and interchange with a one-sided exact McNemar over the same pairs.

**Planned but not run:** R3-steer (geodesic vs linear-chord steering) and R3-sae (SAE features; no SAE exists for post-trained Qwen3.5-2B). Both are in the README to-do.

## Results in words
Numbers are in [`results/RESULTS.md`](../../../results/RESULTS.md), section "Stage 3".

1. **Behaviour.**
   - The model picks the tree-closer animal in somewhat more than half of the within-stratum triplets (binomial p < 0.05, but the triplets are not independent).
   - Its answers agree with its layer-6 representation more often than if the two were independent.
   - On the conflict triplets it picks the tree-closer animal in only a minority of cases, as its representation does.
   - Class-word accuracy is high overall but low for echinoderms and molluscs.
2. **C-abl: not supported.**
   - Ablating the probe subspace lowers log P(correct class) more than the *average* random subspace. This is why the secondary per-species Wilcoxon test is significant.
   - The random subspaces have a heavy tail, though: about one in nine caused a drop at least as large as the probe's, so the registered rank p is far above the Holm threshold.
3. **C-swap: supported, but small.**
   - Swapping the probe coordinates switches the class answer more often than any of the 99 random subspaces. The rank p is at its design floor of 1/100, and the result survives Holm correction.
   - In absolute terms only a few percent of pairs switch, against a few without any edit. The subspace moves the class decision but does not by itself determine it.
4. **Off-target.** Ablating the probe subspace does not raise neutral-text loss; it is marginally lower than without an edit. Random ablations raise it slightly. The subspace is therefore not generally needed for language modelling at this layer.
5. **Class-only control: no phylogeny-specific role.**
   - The class-only subspace produces a *larger* ablation drop than the phylogeny probe and about the same switch rate. Both are well beyond the random subspaces.
   - The phylogeny probe beats it on neither test, so the registered rule is not met.
   - What the interventions show: the model uses this region of activation space for coarse class answers. They do not show that it uses within-group phylogenetic structure.
6. **First run vs final run.**
   - The probe's ablation drop is identical in both runs, since it is the same subspace.
   - The mean random drop is higher in the final run. Its random subspaces are matched to the probe's variance on the *training* species, which is higher than the all-species value the first run used. The first run under-matched its random controls and so made the probe look relatively stronger.
   - The swap rates differ slightly because the final run draws its 200 pairs from a separate seeded stream.
   - The first run's rank p-values could not pass Holm by construction (floors 0.048 and 0.091).
   - Both runs are reported side by side, as deviation 20 requires.

## Interpretation
- **Causal claim (narrow, intervention).** Interchanging the probe subspace changes the model's class answer more than random subspaces do. This is evidence that the subspace is *used*, at layer 6, for coarse classification.
- **No claim of phylogeny-specificity.** A probe that knows only the 13 coarse groups does at least as well. This matches stage 2, where the probe's all-pairs correlation mostly reflects coarse groups.
- **What a within-class causal test would need.** A readout that depends on within-class relatedness (e.g. "Which is more closely related to a lion: a tiger or a wolf?") and a subspace trained on within-stratum distances only. This is listed as future work.

## Caveats
- **Readout.** The readout is the class word, so these tests cannot probe within-class phylogeny directly.
- **Variance mismatch.** The class-only subspace captures more variance than the phylogeny probe and was not variance-matched to it. Its larger ablation drop may partly reflect this. The registered rule asks only whether the phylogeny probe beats it, and it does not.
- **Small counts.** The swap result rests on small counts over 200 pairs. The rank p-value treats the 99 random subspaces as the null and ignores uncertainty from pair sampling.
- **One layer, one probe split.** Effects at other layers and other family splits were not tested.
- **Disclosed forking path.** The decision to rerun with 99 subspaces was made after a favourable first run, because the first design could not pass its own threshold. Deviation 20 discloses this, and both runs are reported.
- **Random subspaces.** They are restricted to the top-64 principal components. This makes them comparable in variance, but they do not span the whole residual stream.

## Audit note (2026-11-22)
From the [independent audit](../../reviews/2026-11-22-independent-audit.md); numbers in `results/qwen35_2b/SUMMARY.md`, section "Independent audit":
- **Variance matching checked on the held-out species.** The interventions act on held-out species, but the random subspaces were matched on training species. The random subspaces were redrawn exactly (same seed and fold) and measured on the held-out species: they lose about as much variance there as the probe does (the probe's held-out share is bounded from its stored all-species and training shares). The final run is therefore roughly matched where the interventions act. The first run's random subspaces were smaller, so "the first run under-matched its random controls" holds.
- **Registered numbers re-derived.** The rank p-values, Holm adjustment and the exact McNemar p were recomputed from the stored per-subspace values and agree.
