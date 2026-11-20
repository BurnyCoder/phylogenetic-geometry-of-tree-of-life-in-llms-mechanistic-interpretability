# Red-team review: stage-3 causal results (2026-11-20)

**Scope.** The final causal run (deviations 19–20), as written up in [experiment 04](../experiments/04-causal-behavior-stage3/experiment.md), the paper section "Behaviour and causal tests", the README and the summary.

**Method.**
- Every claim in those texts was checked against `results/qwen35_2b/stage3/results.json` and the code: `stage3.py` (`run_stage3`) and `causal.py` (`ablate_fn`, `swap_fn`, `random_subspaces`, `probe_subspace`).
- The counts quoted below come from the stored arrays `random_mean_drops` and `switch_rate_random`, read with numpy.

## Verdict
The write-up states the registered outcome correctly:
- C-swap is supported after Holm correction.
- C-abl is not supported.
- The phylogeny-specific rule from deviation 20 is not met.

No claim needs to be withdrawn. Points 1–3 are the strongest objections; each is already stated as a caveat.

## Objections and checks
1. **The C-swap effect is small in absolute terms.**
   - *Counts.* The probe switches 11 of 200 pairs. Without an edit, 4 switch. The best of the 99 random subspaces switches 8.
   - *What the test says.* The rank test is valid for "more than variance-matched random subspaces". Its p-value ignores pair-sampling noise, though, and a difference of 3 pairs over the best random subspace is fragile.
   - *Mitigation.* The superseded first run used a different draw of 200 pairs and also placed the probe above every random subspace, which is weak evidence that the ordering is not a fluke of one pair draw.
   - *Status:* stated ("small effect", "small counts"). Adequate.
2. **The secondary Wilcoxon test is significant while the primary rank test is not.**
   - *Why.* The random-subspace ablation drops are heavy-tailed: 11 of 99 reach the probe's drop, while their median is far below it. The Wilcoxon compares each species with the *mean* random drop, so it ignores that spread.
   - *Status:* explained in experiment 04 and the paper. Deviation 19 registered the rank test as primary before the rerun. Adequate.
3. **The class-only control is not variance-matched to the probe.** It captures more variance (14.4% vs 12.3% on the training species).
   - *What this rules out.* A claim that the class subspace is *more* causal than the phylogeny probe. The texts make no such claim.
   - *What it does not affect.* The registered rule asks whether the phylogeny probe beats the class subspace, and it does not; the mismatch favours the class subspace on ablation but cannot turn a probe deficit into a win.
   - *Status:* stated in the caveats and the paper's limitations. Adequate.
4. **Variance matching of the random subspaces.**
   - *What was checked.* The random subspaces are drawn inside the span of the top-64 PCs of the training species, and on average they capture slightly more variance than the probe.
   - *Why it matters.* A larger random control makes the test conservative, not liberal.
   - *Status:* correct as implemented.
5. **Could the swap work by pushing activations off-distribution?** The random subspaces have matched variance and pass through the same edit, so a generic off-distribution effect would affect them too. Not a concern for the registered comparison.
6. **The readout cannot see within-class phylogeny.**
   - *Why.* The answer is a class word, so a perfect phylogeny subspace and a pure class subspace predict the same outcome.
   - *Consequence.* The test can falsify "phylogeny-specific use" only weakly. The texts claim class use only.
   - *Status:* stated. A within-class readout is listed as future work.
7. **Forking path.**
   - *What happened.* The rerun was decided after a favourable first run.
   - *Why it is acceptable.* The first design could not pass its own threshold, the rerun's settings were fixed in deviations 19–20 before it started, and both runs are reported.
   - *Effect of the rerun.* It moved the result *against* the method (C-abl lost significance), which argues against motivated reanalysis.
8. **Off-target loss.** Ablating the probe subspace slightly *lowers* neutral-text loss, by under 1%. This is too small to interpret. The texts only say that loss does not rise.
9. **Multiple testing.** The two class-control tests are in the BH exploratory family, which grows from 176 to 178 tests. Their q-values are in `results/RESULTS.md`.

## Wording checks
- **"Swapping … more often than any of 99 random subspaces"** (README). Verified: no random subspace reaches the probe's switch rate.
- **"Ablating it does not beat the random subspaces"** (README). Verified: the rank p-value is above the Holm threshold.
- **"A probe trained only on class membership does as well on both tests".** Verified: the class subspace has a larger ablation drop and the same or a higher switch rate, and both one-sided tests are far from significant.
- **Experiment 04 method text.** It was corrected to match the code:
  - the ablation acts at every name token;
  - the swap acts at B's last name token;
  - the random subspaces are matched on average via m = 64 PCs.
