# Progress

## 2026-11-20 07:30 — setup and preregistration
**Done**
- Literature search across three passes: Goodfire Evo 2 method, the neural-geometry and LLM-geometry literature, and phylogeny data and tooling. Plus an independent design critique.
- `reports/literature-review.md` and `reports/state-of-the-art-review.md` (fact-check pass running).
- `reports/preregistration.md` and `reports/planned-runs.md`, committed before any activation exists.
- Environment set up:
  - uv with Python 3.12;
  - torch 2.15.0+cu132, which supports the RTX 5070's sm_120;
  - transformers 5.23.0.
- Qwen3.5-2B download started into `D:\AI\hf_cache`.

**Now**
- Push the preregistration to GitHub.
- Write the tests, then the core implementation.

**Next**
- Freeze the species set and tree, run the power simulation and the Pythia-410m pilot.
- Then stage 1a: top-3 PCA plots.

**Overall:** about 10% of the plan is done. Estimated 3–5 days of wall-clock time remain.
