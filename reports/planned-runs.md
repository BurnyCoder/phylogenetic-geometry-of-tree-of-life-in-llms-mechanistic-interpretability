# Planned runs (run matrix)

*Libor "Burny" Burian, 2026-11-20. This matrix is fixed by the [preregistration](preregistration.md).*

**What every run shares unless the row says otherwise:**
- the same frozen species set (`data/species.csv`, sha256-hashed);
- the same 8 templates;
- last-name-token readout, per-template mean-centering, and cosine distance;
- all layers.

**What "training" means here.** The only trained components are small probes: the stage 2 flat-subspace probe and the stage 2/3 Poincaré probe. LLM weights are never trained. Every LLM run is inference only: bf16, the whole model on one RTX 5070 Laptop GPU (8 GB).

| ID | Stage | Model(s) | Input / condition | What is computed | Trained? | Est. GPU time |
|---|---|---|---|---|---|---|
| R0 | pilot | Pythia-410m | raw text + BOS | whole pipeline end to end (debug only) | no | ~2 min |
| R00 | power | none (simulation) | Brownian-tree + class-blob synthetic clouds on the frozen tree | minimum detectable TA_within | no | CPU |
| **R1a** | **1a (first)** | **Qwen3.5-2B** | chat template, 400 species × 8 templates | **top-3 PCA plots per layer, tree overlay, H1a checks C1–C4** | no | ~3 min |
| R1a-cal | 1a | Qwen3.5-2B; random-init Qwen3.5-2B; synthetic | days of week, months; same species set | calibration PCA plots (circle expected for days/months) | no | ~2 min |
| **R1b** | **1b** | **Qwen3.5-2B** | chat template | **H1 (TA_within, max-T, cross-fit, bootstrap); S1–S5; depth profile; noise ceiling** | no | analysis only |
| R1b-ctl | 1b | random-init Qwen3.5-2B; Pythia-410m step0; Pythia-410m | same | every H1/S metric | no | ~5 min |
| R1b-base | 1b | none | char-3-gram, input embedding, GloVe, class oracle, rank oracle, class-blob sim | every H1/S metric | no | CPU |
| R1b-conf | 1b | Qwen3.5-2B (+ controls) | 36 conflict triplets (separate name set) | phylogenetic vs folk choice per triplet | no | ~1 min |
| R2-probe | 2 | Qwen3.5-2B | layer selected in R1b | flat-subspace probe (k=10), 5-fold family holdout, 5 seeds; live trackio dashboard | **yes (probe)** | ~10 min |
| R2-ctl | 2 | Qwen3.5-2B (shuffled tree); random-init Qwen; GloVe | same | probe selectivity controls | yes (probe) | ~15 min |
| R2-geom | 2 | Qwen3.5-2B + controls | all layers | geodesic vs angular; δ_rel vs matched-Gaussian and non-hierarchical concept sets; NJ tree vs reference (RF, triplet distance) | no | CPU |
| R3-abl | 3 | Qwen3.5-2B | raw text without the chat template; mean-over-name-tokens readout; raw cosine; z-scored cosine; binomial names; polysemous names included; dated tree; kNN K; probe dimension k ∈ {2,3,5,10,20}; Poincaré vs Euclidean probe | effect on H1/S metrics and probe correlation | probes only | ~30 min |
| R4-sens | 4 | Qwen3.5-2B | 5 probe seeds; 20 random template halves; 20 species subsamples of 80%; family-split seeds | variance of every headline number | probes only | ~30 min |
| R5-scale | 5 | Qwen3.5-0.8B, Qwen3-1.7B; Pythia 70m/160m/410m/1b/1.4b/2.8b | chat template (Qwen); raw text (Pythia) | H1/S1 across scale and architecture | no | ~30 min |
| R5-dyn | 5 | Pythia-1.4B at 12 log-spaced checkpoints (step0 … step143000) | raw text | H1/S1 over training; infini-gram Pile frequency and co-occurrence | no | ~30 min |
| R6-causal | 6 | Qwen3.5-2B | held-out species; probe subspace vs variance-matched random subspace | C-abl (mean-ablation), C-swap (interchange); off-target loss on neutral text | no | ~20 min |
| R6-behav | 6 | Qwen3.5-2B | "Which is more closely related to the X: the Y or the Z?" (both orders, log-prob scoring, thinking off) | behavioral triplet accuracy vs representational TA; conflict triplets | no | ~10 min |
| R6-steer | 6 | Qwen3.5-2B | geodesic vs linear-chord steering between species | change in taxonomic answer distribution | no | ~20 min |
| R6-sae | 6 | Qwen3.5-2B (+ Base SAE transfer check) | SAE features on species activations | features aligned with the probe subspace | no | ~10 min |

**Execution order** (following the user's instructions):
1. R0, then R00.
2. **R1a**: simple PCA first.
3. R1b with controls, baselines and conflict triplets.
4. R2 (trained probes and controls).
5. R3 (ablations).
6. R4 (sensitivity).
7. R5 (more models).
8. R6 (causal and behavioral).

After each stage, results, reports, paper, README and site are updated and pushed to GitHub, and data and probes to Hugging Face.
