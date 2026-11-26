# Summary of phylogenetic geometry in animal names

Libor “Burny” Burian · 26 November 2026. On the frozen English-name panel, the main model resolves within-group triples above chance. Its advantage over both untrained networks depends on the resampling level. The evidence supports a geometric association. A phylogeny-specific causal interpretation remains unestablished. [Paper](paper.pdf) · [Independent reanalysis](reviews/2026-11-26-scientific-audit.md) · [Evidence files](../results/scientific-verification)

### What was tested

Qwen3.5-2B read English animal names in several sentence frames. We took the hidden state at the last token of each name, centered it across species within each frame, and averaged the frames. Three species from one broad group make a test case when the reference tree resolves a sister pair and all three names end in different words. This tests distinctions *within* familiar groups. Goodfire’s work on Evo 2 motivated the graph and subspace comparisons. [Preregistration and amendments](preregistration.md) · [Frozen panel](../data/species.csv) · [Goodfire study](https://www.goodfire.com/research/phylogeny-manifold)

<!-- KEY-RESULTS:START -->
| ID | Hypothesis | Estimate | Test p | Assessment |
|---|---|---|---|---|
| H1 | Within-group rooted triples | 0.473 cross-fitted | 0.000999 | Criterion met with family blocks |
| S1 | Global edge-count association | 0.362 at layer 6 | 0.0050 | Criterion met |
| S2 | Family/genus resolution within order | 0.544 at layer 6 | 0.0050 | Criterion met |
| S3 | Unique wrong spelling choice | 0.382 at layer 6 | 0.2438 | Criterion unmet |
| S4 | Unique wrong GloVe choice | 0.372 at layer 6 | 0.2438 | Criterion unmet |
| S5 | Age given rank and MRCA depth | 0.042 at layer 6 | 0.1409 | Criterion unmet |
| C-abl | Candidate log-probability drop | 0.083 nats | 0.2400 | Criterion unmet (Holm upper bound) |
| C-swap | New donor-class answers | 7–11 / 200 | — | Paired comparison unevaluable |
| M1 | Marine versus ground mammals | 6 marine / 61 ground | 4.81e-07 | Criterion met |
| M2 | Aerial versus ground mammals | 0 aerial | — | Unevaluable |
| B1 | Bird water foraging | ρ = -0.385 | 0.000635 | Criterion met |
| B2 | Bird aerial foraging | ρ = -0.157 | 1.0000 | Criterion unmet |

H1 uses the layer maximum. Secondary and habitat rows report Holm-adjusted p-values. Secondary effects are displayed at the H1 layer; their tests cover all layers. C-swap is bounded by the observed marginals. [Audit, evidence and interpretation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-scientific-audit.md)
<!-- KEY-RESULTS:END -->

### What the comparisons tell us

The model’s advantage survives a test on names with no shared three-character strings. The separate tests where spelling or GloVe positively favor the wrong pair remain inconclusive. Global agreement also appears in simple controls based on broad groups or name popularity. A class label, a rooted triple, and a divergence time are different prediction targets. [Group and control tables](../results/qwen35_2b/SUMMARY.md#stage-1b) · [Dating test](../results/scientific-verification/dating.json)

Coarse groups and a water-foraging gradient are visible in projections. The class-centroid-plus-noise probe outperforms the real-input probe on the all-pairs correlation, and the original vectors outperform the probe within strata on identical held-out pairs. Responses to a class-word intervention therefore leave open whether the model uses fine evolutionary relationships to answer questions. [Geometry and probes](experiments/03-geometry-probes-stage2/experiment.md) · [Behavior and edits](experiments/04-causal-behavior-stage3/experiment.md) · [Habitat study](experiments/06-habitat-addendum/experiment.md)

Changing common names to scientific binomials reduces triplet accuracy at the common-name layer. Changing the species sample, prompts, or triplet seed measures sensitivity of this analysis. Differences along the Pythia training series describe a single trajectory. The new habitat species also overlap the discovery panel at higher taxonomic ranks. These features limit extrapolation to new clades, languages, or independently trained models. [Ablations](experiments/05-ablations-stage4/experiment.md) · [Checkpoints](experiments/07-more-models-stage5/experiment.md) · [Sensitivity](experiments/08-sensitivity-stage6/experiment.md) · [Worked examples](../results/RESULTS.md#nearest-neighbor-examples)
