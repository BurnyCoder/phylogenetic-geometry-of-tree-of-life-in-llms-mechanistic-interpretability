# Results

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


All stage-specific exploratory tables below retain their original scoring and test families unless explicitly identified as corrected.

## Power simulation

Single-layer permutation rejection frequencies, 40 simulations per setting. Intervals describe Monte Carlo uncertainty. Power of the joint H1 rule is unknown.

| signal | mean TA_within | SD | rejection fraction | exact 95% CI |
|---|---|---|---|---|
| 0.0 | 0.323 | 0.019 | 0.00 | [0.000, 0.088] |
| 0.05 | 0.331 | 0.028 | 0.10 | [0.028, 0.237] |
| 0.1 | 0.337 | 0.018 | 0.03 | [0.001, 0.132] |
| 0.15 | 0.348 | 0.021 | 0.20 | [0.091, 0.356] |
| 0.2 | 0.358 | 0.020 | 0.28 | [0.146, 0.439] |
| 0.3 | 0.386 | 0.021 | 0.80 | [0.644, 0.909] |
| 0.5 | 0.472 | 0.022 | 1.00 | [0.912, 1.000] |


## Run `pilot_pythia410m_prefix`

### Stage 1a — top-3 PCA checks per layer

| layer | PC1–3 var | C1 purity (null) p | C1 random-init | C2 mammals (null) p | C2 birds (null) p | C2 insects (null) p | C3 overlay/null p | C3b overlay/within-null p | reading |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 13.6% | 0.39 (0.12) 0.005 | – | 0.34 (0.18) 0.005 | 0.13 (0.11) 0.150 | 0.39 (0.16) 0.005 | 0.60 0.005 | 0.75 0.005 | tree-consistent (checks 1-3 pass) |
| 1 | 17.8% | 0.39 (0.12) 0.005 | – | 0.38 (0.18) 0.005 | 0.13 (0.11) 0.120 | 0.42 (0.17) 0.005 | 0.60 0.005 | 0.74 0.005 | tree-consistent (checks 1-3 pass) |
| 2 | 17.1% | 0.37 (0.12) 0.005 | – | 0.35 (0.18) 0.005 | 0.16 (0.11) 0.020 | 0.40 (0.17) 0.005 | 0.60 0.005 | 0.75 0.005 | tree-consistent (checks 1-3 pass) |
| 3 | 18.6% | 0.36 (0.12) 0.005 | – | 0.36 (0.19) 0.005 | 0.14 (0.10) 0.060 | 0.40 (0.17) 0.005 | 0.63 0.005 | 0.77 0.005 | tree-consistent (checks 1-3 pass) |
| 4 | 17.8% | 0.36 (0.12) 0.005 | – | 0.39 (0.19) 0.005 | 0.16 (0.11) 0.010 | 0.43 (0.16) 0.005 | 0.63 0.005 | 0.77 0.005 | tree-consistent (checks 1-3 pass) |
| 5 | 16.0% | 0.38 (0.12) 0.005 | – | 0.34 (0.18) 0.005 | 0.15 (0.11) 0.035 | 0.48 (0.17) 0.005 | 0.64 0.005 | 0.78 0.005 | tree-consistent (checks 1-3 pass) |
| 6 | 14.6% | 0.43 (0.12) 0.005 | – | 0.38 (0.18) 0.005 | 0.17 (0.11) 0.005 | 0.46 (0.16) 0.005 | 0.66 0.005 | 0.80 0.005 | tree-consistent (checks 1-3 pass) |
| 7 | 14.5% | 0.42 (0.12) 0.005 | – | 0.38 (0.18) 0.005 | 0.16 (0.10) 0.010 | 0.49 (0.17) 0.005 | 0.67 0.005 | 0.82 0.005 | tree-consistent (checks 1-3 pass) |
| 8 | 16.2% | 0.46 (0.12) 0.005 | – | 0.37 (0.18) 0.005 | 0.17 (0.10) 0.015 | 0.49 (0.17) 0.005 | 0.67 0.005 | 0.83 0.005 | tree-consistent (checks 1-3 pass) |
| 9 | 17.0% | 0.46 (0.12) 0.005 | – | 0.40 (0.18) 0.005 | 0.14 (0.11) 0.055 | 0.50 (0.16) 0.005 | 0.67 0.005 | 0.84 0.005 | tree-consistent (checks 1-3 pass) |
| 10 | 16.7% | 0.44 (0.12) 0.005 | – | 0.40 (0.18) 0.005 | 0.12 (0.11) 0.225 | 0.48 (0.16) 0.005 | 0.68 0.005 | 0.85 0.005 | tree-consistent (checks 1-3 pass) |
| 11 | 17.5% | 0.42 (0.12) 0.005 | – | 0.39 (0.18) 0.005 | 0.14 (0.10) 0.045 | 0.48 (0.17) 0.005 | 0.69 0.005 | 0.85 0.005 | tree-consistent (checks 1-3 pass) |
| 12 | 19.8% | 0.44 (0.12) 0.005 | – | 0.43 (0.18) 0.005 | 0.14 (0.11) 0.035 | 0.49 (0.17) 0.005 | 0.67 0.005 | 0.84 0.005 | tree-consistent (checks 1-3 pass) |
| 13 | 20.8% | 0.45 (0.12) 0.005 | – | 0.38 (0.18) 0.005 | 0.15 (0.11) 0.040 | 0.50 (0.17) 0.005 | 0.67 0.005 | 0.85 0.005 | tree-consistent (checks 1-3 pass) |
| 14 | 21.6% | 0.45 (0.12) 0.005 | – | 0.38 (0.19) 0.005 | 0.15 (0.11) 0.015 | 0.51 (0.16) 0.005 | 0.67 0.005 | 0.84 0.005 | tree-consistent (checks 1-3 pass) |
| 15 | 22.1% | 0.53 (0.12) 0.005 | – | 0.33 (0.18) 0.005 | 0.15 (0.10) 0.015 | 0.51 (0.16) 0.005 | 0.60 0.005 | 0.82 0.005 | tree-consistent (checks 1-3 pass) |
| 16 | 22.2% | 0.55 (0.12) 0.005 | – | 0.35 (0.18) 0.005 | 0.15 (0.10) 0.015 | 0.50 (0.17) 0.005 | 0.59 0.005 | 0.82 0.005 | tree-consistent (checks 1-3 pass) |
| 17 | 23.3% | 0.57 (0.12) 0.005 | – | 0.36 (0.18) 0.005 | 0.19 (0.11) 0.005 | 0.50 (0.17) 0.005 | 0.58 0.005 | 0.82 0.005 | tree-consistent (checks 1-3 pass) |
| 18 | 23.7% | 0.55 (0.12) 0.005 | – | 0.40 (0.19) 0.005 | 0.19 (0.11) 0.005 | 0.43 (0.16) 0.005 | 0.59 0.005 | 0.83 0.005 | tree-consistent (checks 1-3 pass) |
| 19 | 24.6% | 0.51 (0.12) 0.005 | – | 0.40 (0.19) 0.005 | 0.19 (0.11) 0.005 | 0.43 (0.16) 0.005 | 0.59 0.005 | 0.83 0.005 | tree-consistent (checks 1-3 pass) |
| 20 | 26.9% | 0.47 (0.12) 0.005 | – | 0.38 (0.18) 0.005 | 0.18 (0.11) 0.005 | 0.46 (0.17) 0.005 | 0.60 0.005 | 0.82 0.005 | tree-consistent (checks 1-3 pass) |
| 21 | 27.2% | 0.42 (0.12) 0.005 | – | 0.35 (0.18) 0.005 | 0.17 (0.10) 0.005 | 0.46 (0.16) 0.005 | 0.63 0.005 | 0.83 0.005 | tree-consistent (checks 1-3 pass) |
| 22 | 30.7% | 0.42 (0.12) 0.005 | – | 0.34 (0.18) 0.005 | 0.18 (0.10) 0.005 | 0.43 (0.17) 0.005 | 0.66 0.005 | 0.83 0.005 | tree-consistent (checks 1-3 pass) |
| 23 | 28.9% | 0.45 (0.12) 0.005 | – | 0.35 (0.18) 0.005 | 0.19 (0.11) 0.005 | 0.47 (0.16) 0.005 | 0.66 0.005 | 0.84 0.005 | tree-consistent (checks 1-3 pass) |
| 24 | 26.6% | 0.48 (0.12) 0.005 | – | 0.35 (0.19) 0.005 | 0.16 (0.11) 0.005 | 0.36 (0.17) 0.005 | 0.65 0.005 | 0.85 0.005 | tree-consistent (checks 1-3 pass) |


### Stage 1b — triplet accuracy and secondary hypotheses

**H1 (primary)** — n = 400 species, 17825 sampled triplets, 11122 within-stratum head-noun-disjoint.

| quantity | value |
|---|---|
| cross-fitted TA_within | 0.406 |
| TA_within at modal layer 8 [95% clade-bootstrap CI] | 0.396 [0.366, 0.476] |
| max-T permutation p (all layers) | 0.010 |
| difference vs char3 [95% CI] | [+0.053, +0.169] |
| difference vs input_embedding [95% CI] | [+0.002, +0.121] |
| criterion: maxT_p<0.05 | yes |
| criterion: ci_lower>1/3 | yes |
| criterion: beats_untrained_and_lexical | yes |
| **H1 supported** | **yes** |
| meaningful effect (≥ 0.40) | yes |

**Secondary hypotheses (Holm-adjusted across S1–S5)**

| id | statistic | value | p | Holm p |
|---|---|---|---|---|
| S1 | Mantel Spearman (cosine vs path length) at modal layer | 0.206 | 0.005 | 0.020 |
| S2 | TA at modal layer (n = 4622 triplets) | 0.476 | 0.005 | 0.020 |
| S3 | TA at modal layer (n = 3022 triplets) | 0.323 | 0.560 | 1.000 |
| S4 | TA at modal layer (n = 6670 triplets) | 0.298 | 0.765 | 1.000 |

**TA_within for every representation** (model and controls at their modal layer; chance = 0.333)

| representation | TA_within | 95% CI | Mantel r (path length) |
|---|---|---|---|
| pilot_pythia410m | 0.396 | [0.366, 0.476] | 0.206 |
| char3 | 0.315 | [0.280, 0.338] | 0.032 |
| glove | 0.399 | [0.350, 0.455] | 0.140 |
| class_oracle | 0.333 | [0.333, 0.333] | 0.512 |
| rank_oracle | 0.637 | [0.605, 0.675] | 0.492 |
| tree_oracle | 1.000 | [1.000, 1.000] | 0.444 |
| input_embedding | 0.359 | [0.307, 0.417] | 0.033 |

**Depth profile** (TA over all triplets by deepest rank shared by the three species)

| representation | none (n=1490) | phylum (n=3010) | class (n=8703) | order (n=4574) | family (n=45) | genus (n=3) |
|---|---|---|---|---|---|---|
| pilot_pythia410m | 0.485 | 0.601 | 0.469 | 0.476 | 0.489 | 0.000 |
| char3 | 0.435 | 0.481 | 0.441 | 0.470 | 0.344 | 0.000 |
| glove | 0.469 | 0.569 | 0.472 | 0.556 | 0.378 | 0.333 |
| class_oracle | 0.489 | 0.627 | 0.333 | 0.333 | 0.333 | 0.333 |
| rank_oracle | 0.854 | 0.799 | 0.700 | 0.659 | 0.548 | 0.333 |
| tree_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| input_embedding | 0.411 | 0.507 | 0.435 | 0.457 | 0.452 | 0.222 |


### Conflict triplets (phylogeny vs folk/lexical)

| representation | layer | chose tree-closer partner | accuracy | binomial p (vs 0.5) | McNemar vs main model: main-only / control-only, p |
|---|---|---|---|---|---|
| pilot_pythia410m | 8 | 10/36 | 0.278 | 0.011 | – |
| char3 | 0 | 9/36 | 0.250 | 0.004 | – |
| glove | 0 | 14/36 | 0.389 | 0.243 | – |

The 36 triplets share anchors and partners (e.g. 'human' is the tree-closer partner several times), so binomial and McNemar p-values treat dependent items as independent; read them as descriptive.


Per triplet (pilot_pythia410m, layer 8; ✓ = representation closer to the tree-closer partner):

| anchor | tree-closer | other | type | pilot_pythia410m | char3 | glove |
|---|---|---|---|---|---|---|
| blue whale | hippopotamus | whale shark | ecology+lexical | ✗ | ✗ | ✗ |
| Nile crocodile | house sparrow | green iguana | folk reptile | ✗ | ✗ | ✗ |
| Atlantic bluefin tuna | human | great white shark | folk fish | ✗ | ✗ | ✗ |
| common starfish | human | common octopus | folk sea creature+lexical | ✗ | ✗ | ✗ |
| Atlantic horseshoe crab | goliath birdeater | blue crab | lexical crab | ✗ | ✗ | ✗ |
| little brown bat | cattle | house mouse | folk flying mouse | ✗ | ✓ | ✗ |
| rock hyrax | African bush elephant | guinea pig | folk rodent-like | ✓ | ✗ | ✓ |
| thylacine | red kangaroo | gray wolf | convergence | ✓ | ✗ | ✓ |
| European eel | human | sea lamprey | folk eel-like | ✗ | ✓ | ✗ |
| western honey bee | American lobster | goliath birdeater | folk bugs | ✓ | ✓ | ✓ |
| sugar glider | red kangaroo | southern flying squirrel | convergence | ✓ | ✓ | ✗ |
| aardvark | African bush elephant | giant anteater | convergence | ✗ | ✗ | ✗ |
| Chinese pangolin | cat | giant anteater | convergence | ✗ | ✗ | ✗ |
| European hedgehog | common shrew | North American porcupine | convergence spines | ✗ | ✗ | ✗ |
| West Indian manatee | African bush elephant | walrus | ecology | ✓ | ✓ | ✓ |
| coelacanth | human | Atlantic bluefin tuna | folk fish | ✓ | ✗ | ✗ |
| Australian lungfish | human | Atlantic salmon | folk fish | ✗ | ✓ | ✓ |
| Komodo dragon | king cobra | Nile crocodile | folk lizard-like | ✗ | ✗ | ✓ |
| acorn barnacle | edible crab | common limpet | folk shell | ✗ | ✓ | ✓ |
| common woodlouse | American lobster | seven-spot ladybird | folk bug | ✗ | ✗ | ✗ |
| elephant shrew | African bush elephant | common shrew | lexical shrew | ✗ | ✓ | ✗ |
| red panda | raccoon | giant panda | lexical panda | ✗ | ✗ | ✗ |
| koala | red kangaroo | brown bear | folk koala bear | ✓ | ✗ | ✓ |
| killer whale | bottlenose dolphin | blue whale | lexical whale | ✗ | ✗ | ✗ |
| Indian flying fox | little brown bat | red fox | lexical fox | ✗ | ✗ | ✗ |
| California sea lion | harbor seal | lion | lexical lion | ✗ | ✗ | ✗ |
| common starfish | purple sea urchin | Atlantic cod | lexical fish | ✓ | ✗ | ✓ |
| moon jellyfish | sea anemone | Atlantic cod | lexical fish | ✓ | ✗ | ✓ |
| silverfish | western honey bee | goldfish | lexical fish | ✗ | ✗ | ✗ |
| guinea pig | house mouse | pig | lexical pig | ✗ | ✗ | ✗ |
| seahorse | Atlantic cod | horse | lexical horse | ✗ | ✗ | ✓ |
| electric eel | channel catfish | European eel | lexical eel | ✗ | ✗ | ✗ |
| spotted hyena | cat | dog | folk dog-like | ✗ | ✗ | ✗ |
| fossa | meerkat | cat | folk cat-like | ✓ | ✗ | ✓ |
| pronghorn | giraffe | impala | folk antelope | ✗ | ✗ | ✓ |
| bottlenose dolphin | cattle | great white shark | ecology | ✗ | ✓ | ✓ |


### Stage 2 — geometry and flat-subspace probe

*(no stage-2 results for pilot_pythia410m_prefix)*


### Stage 3 — behavior and causal tests

*(no stage-3 results for pilot_pythia410m_prefix)*


### Stage 4 — ablations

*(no stage-4 results for pilot_pythia410m_prefix)*


### Addendum A — habitat gradient (held-out species)

*(no habitat-addendum results for pilot_pythia410m_prefix)*


### Stage 6 — sensitivity

*(no stage-6 results for pilot_pythia410m_prefix)*


## Run `qwen35_2b`

### Stage 1a — top-3 PCA checks per layer

| layer | PC1–3 var | C1 purity (null) p | C1 random-init | C2 mammals (null) p | C2 birds (null) p | C2 insects (null) p | C3 overlay/null p | C3b overlay/within-null p | reading |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 13.7% | 0.40 (0.12) 0.001 | 0.39 | 0.34 (0.18) 0.001 | 0.13 (0.10) 0.060 | 0.42 (0.17) 0.001 | 0.72 0.001 | 0.79 0.001 | partial (C1 + one of C2/C3) |
| 1 | 13.7% | 0.39 (0.12) 0.001 | 0.19 | 0.32 (0.18) 0.001 | 0.13 (0.10) 0.065 | 0.40 (0.17) 0.001 | 0.65 0.001 | 0.78 0.001 | partial (C1 + one of C2/C3) |
| 2 | 9.8% | 0.31 (0.12) 0.001 | 0.16 | 0.30 (0.18) 0.001 | 0.15 (0.10) 0.021 | 0.40 (0.17) 0.001 | 0.71 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 3 | 12.5% | 0.47 (0.12) 0.001 | 0.15 | 0.40 (0.18) 0.001 | 0.28 (0.10) 0.001 | 0.44 (0.17) 0.001 | 0.64 0.001 | 0.83 0.001 | tree-consistent (checks 1-3 pass) |
| 4 | 16.7% | 0.74 (0.12) 0.001 | 0.14 | 0.44 (0.18) 0.001 | 0.31 (0.10) 0.001 | 0.50 (0.16) 0.001 | 0.42 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 5 | 16.6% | 0.73 (0.12) 0.001 | 0.14 | 0.39 (0.18) 0.001 | 0.35 (0.10) 0.001 | 0.54 (0.17) 0.001 | 0.42 0.001 | 0.79 0.001 | tree-consistent (checks 1-3 pass) |
| 6 | 16.1% | 0.68 (0.12) 0.001 | 0.14 | 0.38 (0.18) 0.001 | 0.35 (0.10) 0.001 | 0.46 (0.17) 0.001 | 0.44 0.001 | 0.79 0.001 | tree-consistent (checks 1-3 pass) |
| 7 | 15.7% | 0.62 (0.12) 0.001 | 0.14 | 0.37 (0.18) 0.001 | 0.34 (0.10) 0.001 | 0.46 (0.17) 0.001 | 0.48 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 8 | 15.8% | 0.51 (0.12) 0.001 | 0.14 | 0.35 (0.18) 0.001 | 0.32 (0.10) 0.001 | 0.45 (0.16) 0.001 | 0.59 0.001 | 0.83 0.001 | tree-consistent (checks 1-3 pass) |
| 9 | 16.3% | 0.46 (0.12) 0.001 | 0.13 | 0.34 (0.18) 0.001 | 0.29 (0.10) 0.001 | 0.40 (0.17) 0.001 | 0.64 0.001 | 0.85 0.001 | tree-consistent (checks 1-3 pass) |
| 10 | 16.5% | 0.41 (0.12) 0.001 | 0.13 | 0.32 (0.18) 0.001 | 0.25 (0.10) 0.001 | 0.40 (0.17) 0.001 | 0.68 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 11 | 17.4% | 0.34 (0.12) 0.001 | 0.14 | 0.31 (0.18) 0.001 | 0.22 (0.10) 0.001 | 0.43 (0.17) 0.001 | 0.73 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 12 | 17.0% | 0.35 (0.12) 0.001 | 0.13 | 0.32 (0.18) 0.001 | 0.21 (0.10) 0.001 | 0.41 (0.16) 0.001 | 0.71 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 13 | 17.9% | 0.29 (0.12) 0.001 | 0.13 | 0.34 (0.18) 0.001 | 0.20 (0.10) 0.002 | 0.39 (0.16) 0.001 | 0.75 0.001 | 0.87 0.001 | tree-consistent (checks 1-3 pass) |
| 14 | 19.2% | 0.28 (0.12) 0.001 | 0.14 | 0.34 (0.18) 0.001 | 0.18 (0.10) 0.002 | 0.36 (0.17) 0.001 | 0.77 0.001 | 0.87 0.001 | tree-consistent (checks 1-3 pass) |
| 15 | 15.4% | 0.54 (0.12) 0.001 | 0.13 | 0.35 (0.18) 0.001 | 0.28 (0.10) 0.001 | 0.50 (0.17) 0.001 | 0.57 0.001 | 0.82 0.001 | tree-consistent (checks 1-3 pass) |
| 16 | 15.8% | 0.66 (0.12) 0.001 | 0.15 | 0.36 (0.18) 0.001 | 0.31 (0.10) 0.001 | 0.48 (0.16) 0.001 | 0.52 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 17 | 14.8% | 0.66 (0.12) 0.001 | 0.15 | 0.40 (0.18) 0.001 | 0.30 (0.10) 0.001 | 0.45 (0.16) 0.001 | 0.52 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 18 | 14.4% | 0.56 (0.12) 0.001 | 0.13 | 0.42 (0.18) 0.001 | 0.33 (0.10) 0.001 | 0.49 (0.16) 0.001 | 0.55 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 19 | 16.4% | 0.68 (0.12) 0.001 | 0.14 | 0.39 (0.18) 0.001 | 0.31 (0.10) 0.001 | 0.49 (0.17) 0.001 | 0.45 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 20 | 18.4% | 0.71 (0.12) 0.001 | 0.15 | 0.39 (0.18) 0.001 | 0.36 (0.10) 0.001 | 0.51 (0.17) 0.001 | 0.41 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 21 | 18.3% | 0.71 (0.12) 0.001 | 0.14 | 0.45 (0.18) 0.001 | 0.34 (0.10) 0.001 | 0.48 (0.17) 0.001 | 0.41 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 22 | 18.1% | 0.71 (0.12) 0.001 | 0.14 | 0.44 (0.18) 0.001 | 0.33 (0.10) 0.001 | 0.43 (0.16) 0.001 | 0.42 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 23 | 17.4% | 0.68 (0.12) 0.001 | 0.14 | 0.47 (0.18) 0.001 | 0.33 (0.10) 0.001 | 0.41 (0.16) 0.001 | 0.44 0.001 | 0.80 0.001 | tree-consistent (checks 1-3 pass) |
| 24 | 19.8% | 0.59 (0.12) 0.001 | 0.13 | 0.41 (0.18) 0.001 | 0.30 (0.10) 0.001 | 0.39 (0.17) 0.001 | 0.57 0.001 | 0.84 0.001 | tree-consistent (checks 1-3 pass) |


### Stage 1b — triplet accuracy and secondary hypotheses

| Representation | TA | Family 95% CI | Paired Δ, families | Paired Δ, orders |
|---|---|---|---|---|
| Qwen3.5-2B | 0.475 | [0.427, 0.535] | — | — |
| Random Qwen | 0.382 | [0.345, 0.445] | [0.014, 0.163] | [-0.014, 0.122] |
| Pythia-410m step 0 | 0.402 | [0.354, 0.460] | [0.008, 0.146] | [-0.020, 0.124] |
| Character trigrams | 0.332 | [0.296, 0.368] | [0.085, 0.216] | [0.059, 0.192] |
| Input embeddings | 0.354 | [0.293, 0.396] | [0.074, 0.203] | [0.042, 0.189] |
| Name token count | 0.378 | [0.320, 0.405] | [0.054, 0.180] | [0.021, 0.151] |
| Log sitelinks | 0.365 | [0.300, 0.398] | [0.066, 0.198] | [0.038, 0.146] |

| ID | Triplets or pairs | H1-layer effect | Layer-max p | Holm p |
|---|---|---|---|---|
| S1 | All pairs | 0.362 | 0.000999 | 0.0050 |
| S2 | 612 | 0.544 | 0.000999 | 0.0050 |
| S3 | 3029 | 0.382 | 0.1219 | 0.2438 |
| S4 | 6072 | 0.372 | 0.1528 | 0.2438 |
| S5 | 75318 | 0.042 | 0.0470 | 0.1409 |


### Conflict triplets (phylogeny vs folk/lexical)

| Representation | Tree closer | Tied | Other closer | Missing |
|---|---|---|---|---|
| Qwen3.5-2B | 11 | 0 | 25 | 0 |
| Random Qwen | 13 | 0 | 23 | 0 |
| Pythia-410m step 0 | 13 | 0 | 23 | 0 |
| Trained Pythia-410m | 10 | 0 | 26 | 0 |
| Character trigrams | 9 | 13 | 14 | 0 |
| GloVe | 14 | 0 | 22 | 0 |

The full per-item outcomes are in [the source table](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/outcomes.json).


### Stage 2 — geometry and flat-subspace probe

Layer 6 (the stage-1b modal layer).

**Flat-subspace probe** (k = 10, PCA-100 fit in-fold, 5-fold family holdout, mean ± sd over seeds)

| input / target | held-out-family Pearson r (all pairs) | Spearman | within-stratum Spearman (mean over strata) | train–test r | train–train r |
|---|---|---|---|---|---|
| model | 0.749 ± 0.010 | 0.749 | 0.326 | 0.848 | 0.959 |
| shuffled_tree | -0.015 ± 0.013 | -0.014 | -0.063 | 0.170 | 0.562 |
| random_init | 0.016 ± 0.013 | 0.012 | 0.074 | 0.162 | 0.519 |
| glove | 0.405 ± 0.005 | 0.411 | 0.277 | 0.588 | 0.845 |
| class_blob | 0.833 ± 0.006 | 0.832 | 0.008 | 0.893 | 0.958 |

Selectivity (model − shuffled-tree, held-out families): 0.763. ± is the SD over probe seeds (folds and initialization), not a CI.

References without a probe: a predictor that knows only the strata reaches Pearson r = 0.871 with path length over all pairs, so the stratum labels alone yield a high in-sample association (compare the class_blob input row). The raw angular distances reach a within-stratum Spearman of 0.290 (all species, no probe).

**Gromov δ_rel** (lower = more tree-like; 0 for a tree metric)

| representation | δ_rel |
|---|---|
| model | 0.322 |
| spectrum-matched Gaussian (10 draws) | 0.460 ± 0.010 |
| class-blob simulation (10 draws) | 0.188 ± 0.007 |
| tree oracle (path length) | 0.000 |

**Neighbor-joining tree vs reference** (normalized Robinson–Foulds; 0 = identical)

| | RF |
|---|---|
| model | 0.812 |
| label-permuted (20) | 0.988 ± 0.0009 (p = 0.048) |
| class-blob inputs (20; coarse groups only) | 0.954 ± 0.0041 (p = 0.048) |

**kNN geodesic vs angular distance** (at the probe layer; gain on the class-blob null for reference)

| metric | angular | geodesic | gain | class-blob null gain |
|---|---|---|---|---|
| Mantel Spearman vs path length | 0.362 | 0.251 | -0.110 | +0.066 |
| TA_within | 0.471 | 0.444 | -0.027 | +0.002 |

With zero-weight edges preserved at layer zero, minimum connected k changes from 11 to 9. Historical plots use the original graph. The probe's original target scale included held-out species and the layer was selected on the full panel.


### Stage 3 — behavior and causal tests

**Behavior** (answer-letter log-odds, averaged over both answer orders; binomial test vs 0.5, two-sided)


| test | n | accuracy | binomial p |

|---|---|---|---|

| conflict triplets: chose the tree-closer animal | 36 | 0.306 | 0.029 |

| within-stratum species triplets: chose the tree-closer animal | 300 | 0.567 | 0.024 |

| class word correct (argmax over candidate class words) | 385 | 0.909 | – |


On the same species triplets, the layer-6 representation is closer to the tree-closer animal in 0.613 of cases; behavior and representation agree on 0.633.


Class-word accuracy by stratum: amphibians 1.00, arachnids 0.93, birds 0.99, cartilaginous fish 1.00, cnidarians 0.80, crustaceans 0.87, echinoderms 0.50, insects 0.76, mammals 0.97, molluscs 0.65, ray-finned fish 0.96, reptiles 0.97.


| Run | Random spaces (ablation / swap) | Donor before / after | New donor bounds | Donor prevalence rank p |
|---|---|---|---|---|
| Original | 20 / 10 | 4 / 13 | 9–13 | 0.0909 |
| Expanded | 99 / 99 | 4 / 11 | 7–11 | 0.0100 |

The behavioral binomial calculations above assume independent questions. The same species recur.

Mean ablation drop was 0.083 nats for the probe, 0.027 for the random spaces and 0.146 for the class-only subspace. The class-only post-edit donor fraction was 0.060. The original random-swap ranks test post-edit prevalence. The registered switch comparison is unevaluable.


### Stage 4 — ablations

**Single-change ablations** (TA_within on the stage-1b triplets; modal layer 6)

| variant | TA_within (modal layer) | max over layers (layer) | Mantel r (modal) |
|---|---|---|---|
| main (last token, per-template centering) | 0.475 | 0.475 (6) | 0.362 |
| raw cosine (no centering) | 0.424 | 0.446 (9) | 0.293 |
| z-scored dimensions | 0.480 | 0.480 (6) | 0.375 |
| stratum-centered | 0.483 | 0.483 (6) | -0.052 |
| mean over name tokens | 0.399 | 0.425 (4) | 0.234 |
| raw text (no chat template) | 0.482 | 0.482 (6) | 0.366 |
| scientific binomials (genus-split triplets) (common names on the same triplets: 0.530; head-noun-disjoint subset, n = 10950: binomials 0.320 (max over layers 0.380), common names 0.476) | 0.332 | 0.386 (20) | 0.134 |

Polysemous names included (n = 414): TA_within 0.470 (primary set 0.475).


**Probe ablations** (held-out families, mean of 3 seeds)

| probe | held-out r (all pairs) | within-stratum Spearman (mean over strata) | pooled within-group Spearman (Simpson-prone) | train r |
|---|---|---|---|---|
| angular_k2 | 0.631 | 0.166 | 0.117 | 0.780 |
| angular_k3 | 0.716 | 0.290 | 0.230 | 0.904 |
| angular_k5 | 0.737 | 0.314 | 0.285 | 0.938 |
| angular_k10 | 0.750 | 0.353 | 0.351 | 0.959 |
| angular_k20 | 0.753 | 0.370 | 0.345 | 0.966 |
| euclid_k2 | 0.699 | 0.205 | 0.202 | 0.853 |
| euclid_k5 | 0.730 | 0.284 | 0.298 | 0.912 |
| euclid_k10 | 0.736 | 0.353 | 0.342 | 0.932 |
| poincare_k2 | 0.675 | 0.191 | 0.194 | 0.843 |
| poincare_k5 | 0.721 | 0.286 | 0.245 | 0.918 |
| poincare_k10 | 0.728 | 0.343 | 0.309 | 0.932 |
| angular_k10_shuffled_tree | 0.034 | 4.65e-05 | 0.034 | 0.585 |
| angular_k10_class_blob | 0.834 | 0.003 | 0.154 | 0.959 |

**kNN geodesic, k sweep** (smallest connected k = 3)

| k | Mantel r vs path length | TA_within |
|---|---|---|
| 3 | 0.251 | 0.448 |
| 6 | 0.391 | 0.474 |
| 10 | 0.386 | 0.470 |
| 20 | 0.432 | 0.469 |
| 40 | 0.401 | 0.476 |
| angular | 0.362 | 0.475 |


### Addendum A — habitat gradient (held-out species)

| Prediction | Species-level p | Holm (four slots) | Within-order sensitivity, Holm |
|---|---|---|---|
| M1 | 1.2e-07 | 4.81e-07 | 0.0004 |
| M2 | — | — | — |
| B1 | 0.000212 | 0.000635 | 0.0222 |
| B2 | 0.9173 | 1.0000 | 1.0000 |


### Stage 6 — sensitivity

Layer 6; main-run TA_within 0.475. Mean ± SD [2.5th, 97.5th percentile] over repetitions; margin = model minus the best of the untrained and nuisance controls in the same repetition.

| perturbation | repetitions | TA_within | min | margin over best control (min) |
|---|---|---|---|---|
| triplet sampling seed | 20 | 0.474 ± 0.002 [0.469, 0.476] | 0.467 | 0.077 (0.070) |
| 80% species subsample | 20 | 0.475 ± 0.019 [0.442, 0.511] | 0.437 | 0.084 (0.044) |
| template halves (4 of 8) | 20 | 0.475 ± 0.004 [0.469, 0.481] | 0.469 | – (best layer: 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 22) |
| leave one template out | 8 | 0.478 ± 0.005 [0.471, 0.483] | 0.471 | – (best layer: 6, 6, 6, 6, 6, 6, 6, 6) |
| single template | 8 | 0.467 ± 0.006 [0.458, 0.475] | 0.456 | – (best layer: 6, 6, 22, 6, 24, 22, 6, 6) |
| cross-fit family-split seed | 20 | 0.472 ± 0.004 [0.466, 0.478] | 0.466 | – (modal layer: 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 21, 6, 7, 6, 6, 6) |

**Triplets per decision node K** (one draw each)

| K | within triplets | qwen35_2b | qwen35_2b_random | pythia410m_step0 | char3 | token_count | log_popularity |
|---|---|---|---|---|---|---|---|
| 50 | 3710 | 0.472 | 0.380 | 0.398 | 0.337 | 0.381 | 0.368 |
| 100 | 6466 | 0.476 | 0.384 | 0.396 | 0.331 | 0.389 | 0.366 |
| 200 | 11097 | 0.476 | 0.382 | 0.395 | 0.333 | 0.381 | 0.363 |
| 400 | 18631 | 0.475 | 0.385 | 0.397 | 0.332 | 0.379 | 0.369 |

**Flat probe, 20 fresh seeds** (folds and init): held-out Pearson r 0.750 ± 0.012 [0.728, 0.768]; within-stratum Spearman (mean over strata) 0.359 ± 0.054 [0.279, 0.459]; pooled within-group Spearman (Simpson-prone) 0.338 ± 0.033 [0.281, 0.399].

Mantel r vs path length under 80% species subsamples: 0.362 ± 0.009 [0.349, 0.376]; under template halves: 0.361 ± 0.006 [0.349, 0.368].



## Run `qwen35_2b_prefix`

### Stage 1a — top-3 PCA checks per layer

| layer | PC1–3 var | C1 purity (null) p | C1 random-init | C2 mammals (null) p | C2 birds (null) p | C2 insects (null) p | C3 overlay/null p | C3b overlay/within-null p | reading |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 14.2% | 0.39 (0.12) 0.001 | 0.38 | 0.33 (0.18) 0.001 | 0.14 (0.11) 0.046 | 0.44 (0.17) 0.001 | 0.71 0.001 | 0.79 0.001 | tree-consistent (checks 1-3 pass) |
| 1 | 14.1% | 0.38 (0.12) 0.001 | 0.17 | 0.32 (0.18) 0.001 | 0.15 (0.11) 0.018 | 0.39 (0.17) 0.001 | 0.63 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 2 | 10.0% | 0.31 (0.12) 0.001 | 0.17 | 0.34 (0.18) 0.001 | 0.15 (0.11) 0.019 | 0.41 (0.16) 0.001 | 0.68 0.001 | 0.80 0.001 | tree-consistent (checks 1-3 pass) |
| 3 | 12.6% | 0.47 (0.12) 0.001 | 0.15 | 0.40 (0.18) 0.001 | 0.27 (0.11) 0.001 | 0.45 (0.17) 0.001 | 0.64 0.001 | 0.83 0.001 | tree-consistent (checks 1-3 pass) |
| 4 | 16.6% | 0.75 (0.12) 0.001 | 0.12 | 0.49 (0.18) 0.001 | 0.31 (0.10) 0.001 | 0.51 (0.16) 0.001 | 0.42 0.001 | 0.80 0.001 | tree-consistent (checks 1-3 pass) |
| 5 | 16.5% | 0.72 (0.12) 0.001 | 0.14 | 0.43 (0.18) 0.001 | 0.35 (0.11) 0.001 | 0.54 (0.17) 0.001 | 0.42 0.001 | 0.79 0.001 | tree-consistent (checks 1-3 pass) |
| 6 | 16.1% | 0.68 (0.12) 0.001 | 0.14 | 0.40 (0.18) 0.001 | 0.35 (0.11) 0.001 | 0.46 (0.17) 0.001 | 0.44 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 7 | 15.8% | 0.64 (0.12) 0.001 | 0.13 | 0.38 (0.18) 0.001 | 0.33 (0.11) 0.001 | 0.46 (0.17) 0.001 | 0.47 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 8 | 15.7% | 0.56 (0.12) 0.001 | 0.14 | 0.35 (0.18) 0.001 | 0.31 (0.11) 0.001 | 0.45 (0.16) 0.001 | 0.54 0.001 | 0.82 0.001 | tree-consistent (checks 1-3 pass) |
| 9 | 16.2% | 0.49 (0.12) 0.001 | 0.14 | 0.36 (0.18) 0.001 | 0.28 (0.11) 0.001 | 0.41 (0.17) 0.001 | 0.62 0.001 | 0.85 0.001 | tree-consistent (checks 1-3 pass) |
| 10 | 16.4% | 0.44 (0.12) 0.001 | 0.13 | 0.32 (0.18) 0.001 | 0.26 (0.11) 0.001 | 0.42 (0.17) 0.001 | 0.66 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 11 | 17.2% | 0.36 (0.12) 0.001 | 0.15 | 0.31 (0.18) 0.001 | 0.21 (0.10) 0.001 | 0.37 (0.17) 0.001 | 0.72 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 12 | 16.8% | 0.36 (0.12) 0.001 | 0.14 | 0.33 (0.18) 0.001 | 0.19 (0.11) 0.001 | 0.41 (0.17) 0.001 | 0.70 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 13 | 17.7% | 0.30 (0.12) 0.001 | 0.15 | 0.33 (0.18) 0.001 | 0.18 (0.11) 0.002 | 0.40 (0.17) 0.001 | 0.75 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 14 | 18.8% | 0.27 (0.12) 0.001 | 0.15 | 0.35 (0.18) 0.001 | 0.16 (0.11) 0.006 | 0.35 (0.17) 0.001 | 0.77 0.001 | 0.86 0.001 | tree-consistent (checks 1-3 pass) |
| 15 | 15.4% | 0.55 (0.12) 0.001 | 0.13 | 0.37 (0.18) 0.001 | 0.28 (0.11) 0.001 | 0.50 (0.17) 0.001 | 0.55 0.001 | 0.82 0.001 | tree-consistent (checks 1-3 pass) |
| 16 | 15.7% | 0.70 (0.12) 0.001 | 0.14 | 0.36 (0.18) 0.001 | 0.29 (0.11) 0.001 | 0.48 (0.17) 0.001 | 0.49 0.001 | 0.81 0.001 | tree-consistent (checks 1-3 pass) |
| 17 | 14.9% | 0.69 (0.12) 0.001 | 0.13 | 0.42 (0.18) 0.001 | 0.29 (0.11) 0.001 | 0.44 (0.17) 0.001 | 0.47 0.001 | 0.79 0.001 | tree-consistent (checks 1-3 pass) |
| 18 | 14.4% | 0.61 (0.12) 0.001 | 0.14 | 0.44 (0.18) 0.001 | 0.33 (0.11) 0.001 | 0.47 (0.17) 0.001 | 0.52 0.001 | 0.79 0.001 | tree-consistent (checks 1-3 pass) |
| 19 | 16.6% | 0.67 (0.12) 0.001 | 0.14 | 0.43 (0.18) 0.001 | 0.33 (0.11) 0.001 | 0.48 (0.16) 0.001 | 0.45 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 20 | 18.6% | 0.71 (0.12) 0.001 | 0.15 | 0.43 (0.18) 0.001 | 0.35 (0.11) 0.001 | 0.51 (0.17) 0.001 | 0.41 0.001 | 0.78 0.001 | tree-consistent (checks 1-3 pass) |
| 21 | 18.5% | 0.72 (0.12) 0.001 | 0.15 | 0.48 (0.18) 0.001 | 0.35 (0.11) 0.001 | 0.49 (0.17) 0.001 | 0.40 0.001 | 0.77 0.001 | tree-consistent (checks 1-3 pass) |
| 22 | 18.3% | 0.71 (0.12) 0.001 | 0.16 | 0.49 (0.18) 0.001 | 0.34 (0.11) 0.001 | 0.44 (0.17) 0.001 | 0.42 0.001 | 0.77 0.001 | tree-consistent (checks 1-3 pass) |
| 23 | 17.6% | 0.69 (0.12) 0.001 | 0.14 | 0.50 (0.18) 0.001 | 0.34 (0.11) 0.001 | 0.44 (0.17) 0.001 | 0.44 0.001 | 0.79 0.001 | tree-consistent (checks 1-3 pass) |
| 24 | 19.6% | 0.60 (0.12) 0.001 | 0.12 | 0.44 (0.18) 0.001 | 0.32 (0.11) 0.001 | 0.42 (0.17) 0.001 | 0.55 0.001 | 0.83 0.001 | tree-consistent (checks 1-3 pass) |


### Stage 1b — triplet accuracy and secondary hypotheses

*(no stage-1b results for qwen35_2b_prefix)*


### Conflict triplets (phylogeny vs folk/lexical)

*(no conflict results for qwen35_2b_prefix)*


### Stage 2 — geometry and flat-subspace probe

*(no stage-2 results for qwen35_2b_prefix)*


### Stage 3 — behavior and causal tests

*(no stage-3 results for qwen35_2b_prefix)*


### Stage 4 — ablations

*(no stage-4 results for qwen35_2b_prefix)*


### Addendum A — habitat gradient (held-out species)

*(no habitat-addendum results for qwen35_2b_prefix)*


### Stage 6 — sensitivity

*(no stage-6 results for qwen35_2b_prefix)*


## Stage 5 — more models (scale and training dynamics)

| Archived run | Nominal parameters | Cross-fitted TA | Modal layer | Layer-max p |
|---|---|---|---|---|
| pythia14b | 1.40B | 0.446 | 24 | 0.0050 |
| pythia14b_step0 | 1.40B | 0.347 | 7 | 0.1800 |
| pythia14b_step1 | 1.40B | 0.347 | 7 | 0.1800 |
| pythia14b_step1000 | 1.40B | 0.385 | 24 | 0.0050 |
| pythia14b_step143000 | 1.40B | 0.446 | 24 | 0.0050 |
| pythia14b_step16000 | 1.40B | 0.415 | 9 | 0.0200 |
| pythia14b_step3000 | 1.40B | 0.390 | 10 | 0.0200 |
| pythia14b_step33000 | 1.40B | 0.436 | 16 | 0.0050 |
| pythia14b_step512 | 1.40B | 0.337 | 3 | 0.3400 |
| pythia14b_step64 | 1.40B | 0.340 | 8 | 0.1550 |
| pythia14b_step66000 | 1.40B | 0.431 | 12 | 0.0050 |
| pythia14b_step8 | 1.40B | 0.333 | 13 | 0.4650 |
| pythia14b_step8000 | 1.40B | 0.387 | 10 | 0.0400 |
| pythia160m | 0.16B | 0.373 | 2 | 0.0300 |
| pythia1b | 1.00B | 0.419 | 12 | 0.0100 |
| pythia28b | 2.80B | 0.470 | 18 | 0.0050 |
| pythia410m | 0.41B | 0.416 | 17 | 0.0050 |
| pythia70m | 0.07B | 0.379 | 4 | 0.0100 |
| qwen35_08b | 0.85B | 0.432 | 19 | 0.0050 |
| qwen35_2b | 2.27B | 0.480 | 6 | 0.0050 |
| qwen3_17b | 1.70B | 0.465 | 9 | 0.0050 |

[Frozen-data calculation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/scale.json)

## Exploratory tests: Historical BH adjustment over the recorded tests

BH-FDR over all 178 recorded exploratory p-values in the historical reporting family. The 150 H1a checks are summarized (148 with q < 0.05; they are per-layer stage-1a checks, see the stage-1a table).

| test | p | BH q |
|---|---|---|
| NJ RF vs random trees | 0.048 | 0.050 |
| NJ RF vs class-blob trees | 0.048 | 0.050 |
| conflict triplets (representation) | 0.029 | 0.031 |
| conflict triplets (behavior) | 0.029 | 0.031 |
| species triplets (behavior) | 0.024 | 0.026 |
| C-abl probe > class-only subspace | 0.966 | 0.966 |
| Post-edit donor prevalence: probe versus class-only | 0.773 | 0.778 |
| stage 5 max-T pythia14b | 0.005 | 0.006 |
| stage 5 max-T pythia14b_step0 | 0.180 | 0.184 |
| stage 5 max-T pythia14b_step1 | 0.180 | 0.184 |
| stage 5 max-T pythia14b_step1000 | 0.005 | 0.006 |
| stage 5 max-T pythia14b_step143000 | 0.005 | 0.006 |
| stage 5 max-T pythia14b_step16000 | 0.020 | 0.022 |
| stage 5 max-T pythia14b_step3000 | 0.020 | 0.022 |
| stage 5 max-T pythia14b_step33000 | 0.005 | 0.006 |
| stage 5 max-T pythia14b_step512 | 0.340 | 0.346 |
| stage 5 max-T pythia14b_step64 | 0.155 | 0.160 |
| stage 5 max-T pythia14b_step66000 | 0.005 | 0.006 |
| stage 5 max-T pythia14b_step8 | 0.465 | 0.470 |
| stage 5 max-T pythia14b_step8000 | 0.040 | 0.043 |
| stage 5 max-T pythia160m | 0.030 | 0.032 |
| stage 5 max-T pythia1b | 0.010 | 0.011 |
| stage 5 max-T pythia28b | 0.005 | 0.006 |
| stage 5 max-T pythia410m | 0.005 | 0.006 |
| stage 5 max-T pythia70m | 0.010 | 0.011 |
| stage 5 max-T qwen35_08b | 0.005 | 0.006 |
| stage 5 max-T qwen35_2b | 0.005 | 0.006 |
| stage 5 max-T qwen3_17b | 0.005 | 0.006 |

Rankings in this archive do not recover the registered switch comparison. The ordinary behavioral and conflict binomial p-values treat reused species as independent.
