# Experiments overview

*Libor "Burny" Burian. Last updated 2026-11-20.* Numbers live in `results/` and are rendered into [`results/RESULTS.md`](../../results/RESULTS.md) by `python -m phylogeo.run --phases report`. This page explains *what* was run and *why*; it contains no hand-typed results.

## In plain language
We give a language model short sentences that mention an animal ("Here is a fact about the house sparrow."). We record the model's internal numbers at the animal's name: a list of 2,048 numbers per layer. Then we ask whether animals that are close relatives on the tree of life (sparrow and finch) have more similar internal numbers than distant ones (sparrow and shark), even when their names share no words and even within one familiar group (birds only). We also check that a model with random weights, simple spelling similarity, and word co-occurrence statistics do **not** explain the result.

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
| Cross-fitting | Choose the best layer on half of the families, measure on the other half, then swap. An honest, non-cherry-picked estimate |
| Clade-block bootstrap | Resample whole families to get confidence intervals that respect that related species are not independent |
| Mantel test | Correlation between two distance matrices with a permutation null |
| Conflict triplet | Cases where folk intuition and phylogeny disagree (a whale is closer to a hippo than to a whale shark) |

## Models (networks)
| Model | Role | Architecture | Training | Objective / data | Params | Layers × width |
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
- These hyperparameters were fixed in the preregistration, before any results. The dimension k = 10 matches Goodfire's Evo 2 choice; lr, steps and α are standard small-probe settings, and α is small as in Goodfire's note that results are insensitive to it.
- The sensitivity sweep over seeds is stage 6 ([08](08-sensitivity-stage6/experiment.md)); the sweep over k and metric is stage 4 ([05](05-ablations-stage4/experiment.md)).

## Experiments
| # | Experiment | Stage | Status | Details |
|---|---|---|---|---|
| 00 | Pilot on Pythia-410m (whole pipeline end to end) | pilot | done | [00-pilot](00-pilot/experiment.md) |
| 00b | Power simulation on the frozen tree | power | done | [00b-power](00b-power/experiment.md) |
| 01 | Top-3 PCA shape ("do they look like a tree?"), H1a | 1a | done | [01-pca-shape](01-pca-shape/experiment.md) |
| 02 | Triplet accuracy, Mantel, S1–S5, controls, conflict triplets | 1b | done | [02-triplets-stage1b](02-triplets-stage1b/experiment.md) |
| 03 | Geodesic vs angular, δ-hyperbolicity, NJ tree, flat-subspace probe | 2 | done | [03-geometry-probes-stage2](03-geometry-probes-stage2/experiment.md) |
| 04 | Behavioral judgments; causal ablation and interchange of the probe subspace | 3 | done 2026-11-20: behaviour weakly above chance; C-swap supported (small effect), C-abl not; a class-only subspace does as well, so no phylogeny-specific causal role (deviations 19–20) | [04-causal-behavior-stage3](04-causal-behavior-stage3/experiment.md) |
| 05 | Ablations (readout, normalization, format, names, probe variants) | 4 | done | [05-ablations-stage4](05-ablations-stage4/experiment.md) |
| 06 | Habitat axis on held-out species (Addendum A) | addendum | done | [06-habitat-addendum](06-habitat-addendum/experiment.md) |
| 07 | More models: scale, architecture, training dynamics | 5 | done | [07-more-models-stage5](07-more-models-stage5/experiment.md) |
| 08 | Sensitivity (triplet/probe/split seeds, templates, species subsamples) | 6 | done | [08-sensitivity-stage6](08-sensitivity-stage6/experiment.md) |
| – | Output examples (nearest neighbours of rule-chosen species) | report | done | `results/qwen35_2b/examples/neighbors.json`, paper section "Output examples" |

## Methodology choices and why
- **Last name token readout, 8 templates, per-template centering.** This follows Gurnee & Tegmark's entity readout and Goodfire's pooling. Centering removes the shared anisotropic direction, which otherwise dominates cosine similarity ([Ethayarajh 2019](https://aclanthology.org/D19-1006/)).
- **Chat template for Qwen3.5-2B.** The user chose the post-trained model, and the chat format is how it is normally used. Raw text is a stage-4 ablation.
- **Batch size 1.** Exact, reproducible hidden states (see [learnings](learnings.md)).
- **Triplet accuracy as the primary metric.** It needs only the topology, is robust to monotone distortions, and is not dominated by the coarse-class splits that drive global correlations.
- **Within-stratum, head-noun-disjoint triplets.** These rule out the two cheapest explanations, folk categories and shared words, by design.
