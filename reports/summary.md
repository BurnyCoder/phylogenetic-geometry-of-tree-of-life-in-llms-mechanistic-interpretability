# Summary of phylogenetic geometry in animal names

Libor “Burny” Burian · 26 November 2026. On the frozen English-name panel, the main model resolves within-group triples above chance. Its advantage over both untrained networks depends on the resampling level. The evidence supports a geometric association. A phylogeny-specific causal interpretation remains unestablished. [Paper](paper.pdf) · [Independent reanalysis](reviews/2026-11-26-scientific-audit.md) · [Evidence files](../results/scientific-verification)

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
