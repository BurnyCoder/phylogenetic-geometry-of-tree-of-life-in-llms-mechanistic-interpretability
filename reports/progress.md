# Progress

## 2026-11-20 08:18: core pipeline merged; stage 1a running

**Done**
- **Literature:**
  - literature review and state-of-the-art review;
  - an independent fact-check pass; all about 15 corrections applied ([report](reviews/2026-11-20-fact-check-literature.md)).
- **Preregistration:** pushed before any data (commit `f9c19b9`). Ten pre-data deviations are logged at its end, including the stage renumbering the user asked for.
- **Data frozen (sha256 in `data/FREEZE.json`):**
  - 400 species across 13 strata;
  - a 15-name polysemous sensitivity set;
  - 79 conflict-triplet names (36 triplets);
  - Open Tree reference trees.
- **TimeTree:** dated 301 of 390 internal nodes. Dates are cached locally only and not redistributed.
- **Code and tests:** 48 fast tests plus a GPU test. [PR #1](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/pull/1) merged. Implemented: data selection, RefTree/LCA, extraction, triplet accuracy with max-T / cross-fitting / bootstrap, Mantel, baselines, simulations, stage-1a PCA plots and checks, the stage-1b analysis, conflict triplets, stage-2 geometry and probe, and a report generator.
- **Pilot:** Pythia-410m ran end to end (experiment 00).

**Now (in this order, sequentially, because of the Windows commit limit)**
1. Extract Qwen3.5-2B activations: 400 species, days, months, conflict and polysemous names (about 10 minutes).
2. Extract random-init Qwen3.5-2B (about 10 minutes).
3. Power simulation (about 10 minutes).
4. Red-team implementation review (an independent agent is running).

**Next**
- Stage 1a: look at the uncolored plots, write the description, then the colored plots and the H1a checks.
- Then Pythia controls → stage 1b → stage 2 (probes) → stage 3 (causal and behavioral) → stages 4–6.

**Overall:** about 30% of the plan is done. Stage 1 results are expected within about 1 hour. Estimated total remaining: 2–4 days.
