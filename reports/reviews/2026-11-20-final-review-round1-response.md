# Response to the final review, round 1 (2026-11-20)

> Historical review. The [26 November reanalysis](2026-11-26-scientific-audit.md) independently rechecks the original observations and supersedes interpretations of the switch endpoint, secondary tests, tied choices and test families in earlier records.

The review is [2026-11-20-final-review-round1.md](2026-11-20-final-review-round1.md).
- **How the findings were checked.** Every must-fix finding was first reproduced independently, on the CPU, from the stored activations and results before anything was changed. The class-only predictor, the class-blob probe, the pooled-metric artifact, the S2 and conflict comparisons with the untrained controls, and the per-stratum values all reproduced. Small numeric differences came only from different random draws.
- **Where the numbers are.** Below, "now reported" means the number is generated from `results/` by the code named. Values are in [`results/RESULTS.md`](../../results/RESULTS.md); none are typed here.
- **Re-run status.** Stages 1b, 2, 4, 6 and the conflict analysis were re-run with the same seeds. The registered numbers reproduce, and the new controls are added.

## Must fix

| # | Finding | Action | Where |
|---|---|---|---|
| M1 | Flat-probe headline explained by coarse classes | Added the class-blob probe input (stage 2, k = 10, 5 seeds; stage 4, k = 10, 3 seeds) and the class-only predictor ceiling to stage 2. The probe figure now shows a class-only reference. README, summary, paper abstract, Discussion and Conclusion no longer claim that "a linear subspace carries the tree"; they state that the all-pairs correlation mainly reflects coarse groups. | `stage2.py`, `stage4.py`, `report.py` (`stage2_table`, `probe_bar_png`, `key_results_md`), `paper.py`, README |
| M2 | Pooled within-group Spearman inflated (Simpson-type) | New metric: the mean over strata of the per-stratum Spearman on held-out pairs (`probes/flat.py: per_stratum_spearman`), with a unit test showing that a group-mean predictor scores about 0 on it while the pooled metric is inflated. Reported next to the class-blob input and the raw distances without a probe. The pooled metric is kept only as a labelled, deprecated column. | `probes/flat.py`, `tests/test_treeness_probe.py`, stages 2, 4 and 6 |
| M3 | S2 matched by random-init Qwen | The registered verdict ("supported" against chance) is unchanged. A post-hoc table now reports the untrained controls with paired family-block bootstrap CIs on S2, S3 and S4, and the README, summary and paper state that S2 does not exceed the untrained control. | `analysis.py` (`exploratory.posthoc_vs_untrained`), `report.py`, `paper.py` |
| M4 | Conflict result indistinguishable from untrained networks | `run_conflict` now adds an exact McNemar test of the model against every control on the same triplets. The report states that the triplets are dependent, so the p-values are descriptive. The verdict changed to "geometry does not follow the tree here; no evidence the trained model differs from untrained networks". "Ecology wins" and "significantly" are removed. | `analysis.py`, `report.py`, README, summary, paper |
| M5 | Superseded causal numbers labelled "final" | Runs are now labelled by content: "final" only with ≥ 99 random subspaces and a Holm result; otherwise "first run (superseded)", and its invalid off-target losses are no longer printed. Reports and site regenerated. | `report.py: stage3_table`, `key_results_md` |
| M6 | Undeclared readout and probe-refit choices; C-abl tests class information | **Deviation 20** (`ab13f8f`) was committed before the rerun started. It logs the chat-question readout and the probe refit, discloses that the rerun was decided after a favourable first run, and adds a **class-only subspace control**: the same probe trained on the class-oracle distance. A phylogeny-specific causal role is claimed only if the probe beats it on both ablation (Wilcoxon) and interchange (exact McNemar). Variance matching now uses the training species. | `preregistration.md`, `stage3.py`, `causal.py` (`3a44e92`) |
| M7 | Registered BH-FDR never implemented | `report.py: exploratory_bh` applies one BH pass over every registered exploratory test with a p-value. That covers the H1a checks, the neighbour-joining tests, the conflict triplets (representation and behaviour), the behavioural species triplets, the stage-3 class-control tests and every stage-5 max-T. q-values go to `results/<run>/exploratory_bh.json` and RESULTS.md; the README feature list is corrected. | `report.py` |
| M8 | "Mammals or insects" example wrong | Per-stratum TA_within with family-block bootstrap CIs and node counts is now reported (`exploratory.per_stratum_ci95`). README, summary and paper say the signal is clear only in mammals and birds, and mention that arachnids fall below chance. | `analysis.py`, `report.py`, README, paper |

## Should fix

| # | Action |
|---|---|
| S1 | The lexically neutral subset (character 3-grams have no unique closest pair) is now reported, with paired CIs against the untrained controls (post hoc). The H1 wording says "beyond the registered shortcuts", and the "not shown beyond spelling and co-occurrence" caveat is stated next to it. |
| S2 | The class-oracle and popularity Mantel values are printed next to S1 in the README row, the report and the paper. |
| S3 | The habitat axis is now "top late-layer component" with its variance share, using the registered reading ("mixes ecology with taxonomy"). Name cues and the clustering of water foraging within birds are stated as caveats. |
| S4 | Reworded to "no time signal detected; low power by design". The within-stratum bin analysis is listed as future work. |
| S5 | Fair binomial comparator: genus-split *and* head-noun-disjoint triplets for both name forms, plus the best layer for binomials. Wording changed to "much weaker, late-layer only". |
| S6 | Not run in this round. The claim is restricted to "agreement with the tree", and the ecology and body-size confound is listed as a limitation. An EltonTraits trait-distance baseline for mammals and birds is in the README to-do. |
| S7 | The margin over the strongest required control, with its paired CI, is now in the H1 headline row and in stage 1b. |
| S8 | The behavioural agreement is compared with its independence baseline in the headline row; dependence of the triplets is stated. A family-block bootstrap for the behavioural accuracy is listed as future work. |
| S9 | Neighbour-joining is now also compared with NJ trees built from class-blob inputs (20 draws). |
| S10 | p-values at a permutation floor print as the value itself (e.g. 0.001), never "< 0.001". The probe figure has two panels (all pairs, within strata), class-only and raw-distance reference lines, and an explicit "SD over seeds" label. The by-layer line plot now uses the validated palette with the legend outside the plot. CI bands per layer were not added. |
| S11 | "Untrained networks map similar strings to similar vectors" is now stated as a hypothesis, with the counter-observation (they stay above chance on triplets that share no token). |
| S12 | Changed to "never better than the best flat (Euclidean) probe". |

## Minor
- **Stage 5 double count.** The Pythia-1.4B final checkpoint is no longer counted twice in the scale plot.
- **Not changed.** The variance-matching set (fixed in stage 3), the unpinned tokenizer and GloVe downloads, and the stage-2 triplet stream are left as they are; the tokenizer and GloVe point goes in the limitations. The power simulation is not re-run with more simulations: its minimum-detectable-effect value is labelled as resting on 40 simulations per cell.
- **README to-do and "What is new".** Updated.
