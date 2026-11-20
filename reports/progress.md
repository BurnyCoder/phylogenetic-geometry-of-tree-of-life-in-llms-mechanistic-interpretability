# Progress

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
