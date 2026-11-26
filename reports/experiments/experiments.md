# Experiments overview

Libor “Burny” Burian · 26 November 2026. Each row links to the original experiment. Current findings are generated from saved evidence. [Scientific audit](../reviews/2026-11-26-scientific-audit.md) · [Results](../../results/RESULTS.md)

## Key terms
| Term | Meaning |
|---|---|
| Residual stream / hidden state | The vector a transformer carries from layer to layer at each token; layer 0 is the token embedding |
| Readout | Which token's vector we take: here the **last token of the species name** |
| Species vector | Readout averaged over 8 prompt templates, after subtracting the mean over species within each template |
| Cosine distance | 1 − cosine similarity between two species vectors |
| Reference tree | Open Tree of Life synthetic tree restricted to our species (topology only) |
| Rooted triplet | Three species (a, b, c) where the tree says a and b share a more recent ancestor than either does with c |
| Triplet accuracy (TA) | Fraction of triplets where the representation also puts a and b closest. Chance = 1/3. Every branching point of the tree counts equally ("node-balanced") |
| TA_within | TA over triplets whose 3 species are in the same coarse group (stratum, e.g. birds) and whose names share no head noun. **Primary metric** |
| Stratum | One of 13 coarse groups (mammals, birds, reptiles, amphibians, ray-finned fish, cartilaginous fish, insects, arachnids, crustaceans, molluscs, cnidarians, echinoderms, other) |
| Max-T permutation test | Shuffles which species vector belongs to which tree leaf (within strata) and compares the best layer's TA with the best layer's TA under shuffling. This corrects for having tried 25 layers |
| Cross-fitting | Choose the best layer on half of the families, measure on the other half, then swap. A split evaluation conditional on the panel and chosen analysis |
| Clade-block bootstrap | Resample whole families to get confidence intervals under a family-block approximation. Dependence may also extend across families |
| Mantel test | Correlation between two distance matrices with a permutation null |
| Conflict triplet | Cases where folk intuition and phylogeny disagree (a whale is closer to a hippo than to a whale shark) |

## Models (networks)
| Model | Role | Architecture | Training | Objective / data | Approximate scale-series size label | Layers × width |
|---|---|---|---|---|---|---|
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (rev `15852e8`) | **primary** | hybrid decoder: 3 Gated-DeltaNet linear-attention blocks per full-attention block | pretrained, then post-trained by Qwen (released model, not trained by us) | next-token prediction on large web/code/multilingual corpora, then instruction/chat post-training | 2.27B | 24 × 2048 |
| Qwen3.5-2B random init | untrained control | same | none (random weights, seed 0) | – | 2.27B | 24 × 2048 |
| [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m) | pilot; second trained model | GPT-NeoX decoder | pretrained by EleutherAI | next-token prediction on the Pile ([Biderman et al. 2023](https://arxiv.org/abs/2304.01373)) | 0.41B | 24 × 1024 |
| Pythia-410m `step0` | second untrained control | same | none (checkpoint at step 0) | – | 0.41B | 24 × 1024 |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (rev `2fc0636`) | stage 5: scale within family | hybrid, as Qwen3.5-2B | released post-trained model | as Qwen3.5-2B | 0.85B | 24 × 1024 |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (rev `70d244c`) | stage 5: standard full-attention architecture | decoder, full attention only | released post-trained model | next-token prediction, then post-training | 1.7B | 28 × 2048 |
| Pythia 70m / 160m / 1b / 1.4b / 2.8b | stage 5: scale on one training corpus | GPT-NeoX decoder | pretrained by EleutherAI (final checkpoint) | the Pile | 0.07–2.8B | 6 × 512 … 32 × 2560 |
| Pythia-1.4B at 12 checkpoints (steps 0 … 143000) | stage 5: training dynamics | GPT-NeoX decoder | EleutherAI checkpoints | the Pile | 1.4B | 24 × 2048 |

**Trained by us.** Only small probes: the stage-2 flat-subspace probe, a 10 × 100 linear map plus a scale, trained with Adam (lr 1e-3, 2,000 full-batch steps, α = 0.01), with 5-fold family-holdout CV and 5 seeds.
The size column follows the original scale-analysis metadata. The study used text inputs and language-layer readouts. Published multimodal checkpoint sizes include additional weights. The configs and dated amendments give the training settings and revisions. [Scale metadata](../../results/stage5/summary.json) · [Qwen checkpoint configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)
- The sensitivity sweep over seeds is stage 6 ([08](08-sensitivity-stage6/experiment.md)); the sweep over k and metric is stage 4 ([05](05-ablations-stage4/experiment.md)).

## Experiments
| # | Experiment | Stage | Status | Details |
|---|---|---|---|---|
| 00 | Pilot on Pythia-410m (whole pipeline end to end) | pilot | done | [00-pilot](00-pilot/experiment.md) |
| 00b | Power simulation on the frozen tree | power | done | [00b-power](00b-power/experiment.md) |
| 01 | Top-3 PCA shape ("do they look like a tree?"), H1a | 1a | done | [01-pca-shape](01-pca-shape/experiment.md) |
| 02 | Triplet accuracy, Mantel, S1–S5, controls, conflict triplets | 1b | done | [02-triplets-stage1b](02-triplets-stage1b/experiment.md) |
| 03 | Geodesic vs angular, δ-hyperbolicity, NJ tree, flat-subspace probe | 2 | done | [03-geometry-probes-stage2](03-geometry-probes-stage2/experiment.md) |
| 04 | Behavioral judgments and interventions | 3 | archived, registered switch comparison unevaluable | [04-causal-behavior-stage3](04-causal-behavior-stage3/experiment.md) |
| 05 | Ablations (readout, normalization, format, names, probe variants) | 4 | done | [05-ablations-stage4](05-ablations-stage4/experiment.md) |
| 06 | Habitat axis on held-out species (Addendum A) | addendum | done | [06-habitat-addendum](06-habitat-addendum/experiment.md) |
| 07 | More models: scale, architecture, training dynamics | 5 | done | [07-more-models-stage5](07-more-models-stage5/experiment.md) |
| 08 | Sensitivity (triplet/probe/split seeds, templates, species subsamples) | 6 | done | [08-sensitivity-stage6](08-sensitivity-stage6/experiment.md) |
| – | Nearest neighbors of the most-linked species in each stratum | report | archived | [All examples and comparators](../../results/RESULTS.md#nearest-neighbor-examples) |
| 09 | Independent reanalysis of archived measurements | verification | complete | [09-independent-verification](09-independent-verification/experiment.md) |

## Independent verification

[Experiment and original-data inputs](09-independent-verification/experiment.md) · [Learnings](09-independent-verification/learnings.md) · [Original chronological overview](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/experiments.md)

## In plain language

In plain language, putting nearly all birds together can produce a high global tree correlation even if the model treats relatives within birds indiscriminately. For each of three species from the same broad group, the tree identifies the pair that shares a more recent common ancestor. The primary question asks whether the model places that pair closest together. A strictly increasing transformation of representation distances preserves their ordering. Giving each eligible branching point equal weight limits the influence of nodes with many sampled triples. [Reference and scoring](../../src/phylogeo/stats/triplets.py) · [Registration](../preregistration.md)

## Methodology choices and why

The last name token can depend on all preceding parts of a multiword name in a causal decoder. Reading it in several prompts and subtracting the mean over species within each prompt reduces shared template offsets. Comparisons with the mean over name tokens, raw text, and uncentered cosine check the consequences of those choices. Extraction used single prompts after a batch check found differences in the recorded bfloat16 readouts. [Extraction](../../src/phylogeo/extract.py) · [Ablations](05-ablations-stage4/experiment.md) · [Engineering record](learnings.md)

Character trigrams measure overlapping strings, and sitelinks measure prominence across Wikimedia sites. Neither measures Qwen’s exposure to a name during pretraining. The static GloVe comparison tests one form of distributional similarity. The shuffled-tree and class-centroid probe controls ask different questions about what can predict a large all-pairs correlation. [Lexical baselines](../../src/phylogeo/baselines.py) · [Probe experiment](03-geometry-probes-stage2/experiment.md)
