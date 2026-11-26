---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
tags: [interpretability, probing, phylogeny, representation-geometry]
---
# Flat-subspace probe for animal names in Qwen/Qwen3.5-2B

This checkpoint maps the last name-token readout from layer 6 to 10 coordinates. Its target is edge-count path length on the sampled Open Tree topology. Its training-set Pearson correlation is 0.947 across the 400 fit species. [Checkpoint configuration](https://huggingface.co/BurnyCoder/phylogeo-flat-probe-qwen3.5-2b/blob/main/config.json) [Method inspiration](https://www.goodfire.com/research/phylogeny-manifold)

For each template, center the readouts over the reference species and then average templates. The stored parameters implement `x_pca = (x - pca_mean) @ pca_components.T / z_scale` and `z = (x_pca - b) @ W.T`. Convert angular separation to estimated tree units with `exp(log_beta) * arccos(clip(cos(z_i, z_j), -1 + 1e-6, 1 - 1e-6)) / target_scale`. The archive contains the weights and PCA in `probe.safetensors` and in-sample coordinates in `species_coordinates.csv`. [Probe implementation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/src/phylogeo/probes/flat.py)

Separate historical cross-validation runs yielded mean all-pair Pearson r 0.749 for held-out families, 0.833 for a class-centroid-plus-noise input, and -0.015 for shuffled targets. Mean within-stratum held-out Spearman was 0.326 for the probe and 0.401 for raw distances on the same pairs. The historical procedure chose the layer on the full panel and used the full-panel maximum for target scaling. The released checkpoint used all species. Each cross-validation run had its own fits. [Frozen-data calculation](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/geometry.json) [Archived probe outcomes](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/stage2/results.json)

The intervention archive cannot identify the registered comparison of newly acquired donor answers against random subspaces. No model or probe was evaluated afresh in the November 26 reanalysis. [Independent scientific audit](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-scientific-audit.md)
