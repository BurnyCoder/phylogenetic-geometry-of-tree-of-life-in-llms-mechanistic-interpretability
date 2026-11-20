# Response to the implementation review (2026-11-20)

Review: [2026-11-20-implementation-review.md](2026-11-20-implementation-review.md). Every finding below was addressed **before any confirmatory (stage-1b) statistic was computed on Qwen3.5-2B**. The preregistration logs each change as deviations 11–18.

| # | Severity | Finding | Response | Where |
|---|---|---|---|---|
| 1 | critical | H1 drops missing controls silently and adds trained Pythia | Fixed required set: random-init Qwen, Pythia step0, char3, input embedding, plus the nuisance baselines from finding 2. If a model control is missing, `supported = None` and `missing_controls` lists it. Trained Pythia is descriptive only | `analysis.py::run_stage1b` |
| 2 | critical | Token count and popularity alone reach TA_within ≈ 0.39–0.40 | Both 1-D baselines added to criterion 3 (stricter). Added: TA on nuisance-discordant triplets, and a stratum-centered cosine variant with its own max-T | `baselines.scalar_distance`, `analysis.stratum_centered_distances` |
| 3 | major | S2 passes for lexical-only representations | S2 restricted to head-noun-disjoint triplets | `run_stage1b` (`in_order & hd`) |
| 4 | major | S5 used impossible 0-Mya ages | Undated nodes are NaN and their pairs are excluded. Monotonicity is enforced only among dated nodes. 8 tries per node | `tree.date_tree`, `analysis.s5_rank_vs_time` |
| 5 | major | Missing OTT ranks mis-bin primates and coelacanths | NCBI Taxonomy ranks first (resolving merged taxids), OTT as fallback. Every species now has NCBI phylum, order and genus | `data/ncbi.py`, `species.annotate_with_ott` |
| 6 | major | S5 confounds time with topology; null differs from the registered one | S5 = within-rank-bin **partial** Spearman controlling for MRCA topological depth; the raw value is still reported. The null permutes species within strata (documented) | `analysis.s5_rank_vs_time` |
| 7 | major | Point estimate and CI measure different things | Both reported. The "meaningful" label uses the smaller one | `run_stage1b` (`ta_within_conservative`) |
| 8 | major | Two names were garbled | UTF-8 decoding of SPARQL CSV; cache cleared; re-frozen | `webcache.request_text` |
| 9 | major | Unregistered genus-title exclusion | Removed. *Hippopotamus* and *Dugong* are now in the set | `species.common_name` |
| minor | – | Holm family could silently shrink | S5 is required, so a missing or mismatched age file raises an error | `run_stage1b` |
| minor | – | 429 responses could be cached | Never cached | `webcache.request_json` |
| minor | – | GloVe fill for unknown names | Triplets with unknown names are excluded from S4, and a known-names-only GloVe TA is reported | `run_stage1b` |
| minor | – | One shared random stream | One independent stream per component (`_rng(cfg, name)`) | `analysis._rng` |
| minor | – | No check that activations match the species list | `layer_distances` raises an error on mismatch | `phases.layer_distances` |
| minor | – | Stage-1a reading rule looser than registered | C2 must pass in all three zoom strata | `viz/pca.classify_h1a` |
| minor | – | Layer 24 is the final-norm output | Documented. In HF `hidden_states[-1]` follows the final norm; layer 24 is interpreted with that caveat | docs |
| minor | – | Head-noun mask fixed in the permutation null (conservative) | Kept (conservative); documented | – |
| minor | – | Conflict binomial ignores reused species | Documented as a limitation; per-triplet results are reported | – |
| minor | – | Power was not run before pilot activations were viewed | Power re-run on the re-frozen tree before the new extraction | `logs/bg2_power.out` |

Separately, the plots used two near-identical blues for mammals and ray-finned fish. The palette is now validated with the dataviz skill's `validate_palette.js`, using composite color × marker encoding and spotlight small multiples.
