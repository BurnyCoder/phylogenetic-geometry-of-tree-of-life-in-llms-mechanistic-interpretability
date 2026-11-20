# Progress

## 2026-11-20 12:20: stages 1–4 and 6 complete; stage 3 causal rerun and stage 5 running

**Done since 09:30**
- **Stage 2 (exploratory).** The flat probe generalizes to held-out families far above the shuffled-tree and GloVe controls. There is no geodesic advantage, δ_rel lies between the class-blob and Gaussian nulls, and the neighbor-joining tree is closer to the reference than random trees.
- **Stage 3.**
  - Behavior: the model's relatedness judgments are slightly above chance and agree modestly with its representation.
  - Causal: the first run had too few random subspaces to ever pass Holm, and a mismatched off-target baseline. Both were fixed (deviation 19, committed before the rerun); the rerun is queued after stage 5.
- **Stage 4 (ablations).** The effect is robust to normalization, prompt format and polysemous names. Latin binomials carry no structure. Poincaré is not better than flat. The probe captures within-stratum order.
- **Addendum A (habitat axis, held-out species).** The aquatic end is confirmed; the aerial end is not.
- **Stage 6 (sensitivity).** The model beats the best control on every triplet seed and every 80% species subsample. Templates barely matter, and the cross-fit layer and probe are stable across seeds.
- **Reporting.**
  - Report tables for every stage, with a generated headline block in the README and `reports/summary.md`.
  - The site covers all stages, with an interactive per-layer chart; GitHub Pages is enabled and deploys on merge to main.
  - The paper was rewritten with full sections and references.
  - `docs/pipeline.md`.
- **Hugging Face.** The all-species flat probe is uploaded, and activations of the finished runs are uploading.

**Now (background, one GPU job at a time)**
- **Stage 5.** Finished so far: Qwen3.5-0.8B, Qwen3-1.7B, Pythia 70m/160m/410m/1b. Now running Pythia-1.4b, then 2.8b, then 12 Pythia-1.4B checkpoints, whose downloads are still finishing.
- **Stage 3 causal rerun.** 99 random subspaces for ablation and swap, starting after stage 5. Estimated about 3 hours.

**Next**
- Stage-5 summary and experiment write-up.
- Stage-3 causal results and experiment 04.
- Final fact-check and NeurIPS-style red-team review.
- Merge to main, which deploys the site, then upload the remaining activations.

**Overall:** about 85% done. Remaining wall-clock is dominated by GPU jobs: stage 5 needs about 1.5–2 hours and the stage-3 rerun about 3 hours, checked every 30 minutes.

## 2026-11-20 09:30: stage 1 complete; stage 2 running

**Done**
- **Literature.** Review plus an independent fact-check.
- **Preregistration and review.** Preregistration pushed before data. An independent implementation review found 9 serious issues, all fixed before any confirmatory statistic; deviations 11–18 are logged.
- **Data.** Species set re-frozen: 400 species, NCBI-first ranks, UTF-8 names, and the genus-title rule removed. TimeTree re-dated 308 of 390 internal nodes.
- **Power simulation.** Run on the frozen tree before extraction. Minimum detectable TA_within at 80% power is below the 0.40 smallest effect of interest.
- **Extraction.** Qwen3.5-2B, random-init Qwen3.5-2B, Pythia-410m, Pythia-410m step0.
- **Stage 1a (exploratory).**
  - Class-level arms and within-class order clustering beyond the lexical and random-init references.
  - Days and months form circles (positive control).
  - No nested within-class tree in the top 3 PCs.
  - A late-layer, ecology-like gradient (water → land → air).
- **Stage 1b (confirmatory).**
  - H1 is **supported**, with a meaningful effect: TA_within beats every required control.
  - S1 and S2 are supported.
  - S3 and S4 are not: no evidence beyond spelling or co-occurrence.
  - S5 is not: no divergence-time effect beyond topology.
  - Conflict triplets follow folk categories.
  - Tables: `results/qwen35_2b/SUMMARY.md`.
- **Hugging Face.** Dataset and collection created.

**Now**
- Stage 2: geodesic vs angular, δ-hyperbolicity, neighbor-joining, the flat-subspace probe with controls (running).

**Next**
- Stage 3: causal ablation and interchange of the probe subspace, plus behavioral judgments.
- Stage 4: ablations, including the post-hoc habitat-gradient hypothesis on held-out species.
- Stage 5: more models (Qwen3.5-0.8B, Qwen3-1.7B, Pythia scale and checkpoints).
- Stage 6: sensitivity.
- Then the paper, the site and the final reviews.

**Overall:** about 50% done. Remaining estimate: about 1–2 days, mostly stages 3–6 and the paper.
