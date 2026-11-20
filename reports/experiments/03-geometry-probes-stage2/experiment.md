# Experiment 03 (stage 2): manifold / tree geometry and the trained flat-subspace probe

**Status:** done 2026-11-20.
- **Command:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases stage2
  ```
- **Numbers:** auto-generated in [`results/qwen35_2b/SUMMARY.md`](../../../results/qwen35_2b/SUMMARY.md) (section "Stage 2"). Figures are in `results/qwen35_2b/stage2/`: `probe_heldout_r.png`, `geodesic_vs_angular_*.png`.
- **Live training curves:** the trackio project `phylogeo-probes` (see README).
- **Status of inference:** exploratory (preregistration section 5).

## What was done
1. **Flat-subspace probe** (Goodfire Evo 2 recipe).
   - At the stage-1b modal layer, PCA-100 is fit on training families only. A linear map W (10×100), with offset b and scale β, is trained so that β·arccos(cos(Wx_i, Wx_j)) matches the reference tree's path length (rescaled to [0, π]). There is a small reconstruction term α = 0.01. Training: Adam at lr 1e-3, 2,000 full-batch steps.
   - Evaluation: 5-fold cross-validation holding out **whole families**, 5 seeds.
   - Controls, with the same probe:
     - shuffled-tree target (leaf labels permuted, shape kept);
     - random-init Qwen inputs;
     - GloVe inputs.
2. **kNN geodesic vs angular distance** at every layer, with the smallest k that makes the graph connected (Modell et al. 2025; Goodfire). The gain from geodesic over angular is compared with the same gain on a matched class-blob simulation: stratum centroids plus noise, no within-class tree.
3. **Gromov δ_rel** (Khrulkov et al. 2020 formulation) for the model, compared with a spectrum-matched Gaussian and with the class-blob simulation.
4. **Neighbor-joining tree** from the representation distances, compared with the reference tree by normalized Robinson–Foulds distance, against 20 label permutations.

## Results in words
- **A low-dimensional linear "phylogeny subspace" exists, and it generalizes to unseen families.**
  - Held-out-family distances predicted by the 10-d probe correlate strongly with true tree distance, consistently across seeds.
  - Selectivity over the shuffled-tree control is high.
  - The random-init model gives ≈ 0, so the probe is not fitting noise.
  - GloVe co-occurrence vectors reach about half of the model's held-out correlation.
  - This is the closest analogue to Goodfire's Evo 2 result. The held-out value is somewhat below their reported ~0.89 on held-out clades, but here every control was run.
  - **Caveat:** tree path length is dominated by between-class distances. A class-membership oracle alone has a Mantel correlation of about 0.5 with it (stage 1b table), so part of the probe's correlation is coarse class structure. The probe exceeds that oracle, but a within-class held-out evaluation is still needed and is scheduled for stage 4.
- **No evidence for a curved-manifold advantage.**
  - At the probe layer, geodesic distance on the kNN graph is *worse* than plain angular distance, for both Mantel and TA.
  - At other layers, geodesic gains are small and no larger than those the class-blob null produces by construction.
  - The Goodfire/Modell picture, where the flat angular geometry is a poor proxy for a curved manifold whose geodesics track the tree, is not what we see here.
- **δ-hyperbolicity does not discriminate.** The model is more tree-like than spectrum-matched noise but *less* tree-like than pure class clusters, which score very low δ by themselves. δ_rel alone cannot separate "tree" from "clusters" in this setting, which is the trap the literature review anticipated.
- **The neighbor-joining tree is far from the reference but much closer than chance.** About a fifth of the reference splits are recovered, against ~1% under permutation (p at the 20-permutation floor).

## Interpretation
The strongest stage-2 evidence is the probe. A 10-d linear subspace of layer 6 encodes tree distance well enough to predict it for families never seen in training. This shows that phylogenetic information is **linearly decodable** from the representation, beyond co-occurrence vectors and far beyond untrained networks.

The *raw* geometry (stage 1b) is only moderately tree-aligned, and it is not more tree-like than clustered data on δ. So the tree information is present but mixed with other dominant directions, such as the ecological gradient seen in stage 1a. A trained probe can isolate it.

Whether the model *uses* this subspace is tested causally in stage 3.

## Learnings
- **A probe's high correlation must be benchmarked against the class oracle.** Path length is dominated by deep splits, so "r = 0.75" partly reflects class membership. Always also report within-class held-out performance.
- **Geodesics on smallest-connected kNN graphs can hurt when the data are clustered.** Long bridging edges distort between-cluster distances, and with k = 3 the graph is very sparse.

## Revision after the final review (round 1, 2026-11-20)
The probe interpretation above ("a trained probe can isolate it") is **withdrawn**. See the [review](../../reviews/2026-11-20-final-review-round1.md), M1 and M2.
- **Class-only inputs.** A predictor that knows only the 13 strata already correlates highly with path length over all pairs. The same probe trained on a class-blob input, which has coarse groups and no within-stratum tree, generalizes to held-out families at least as well as the model. So the held-out all-pairs correlation mainly measures coarse-group separation.
- **The within-group metric was flawed.** It pooled all same-stratum pairs into one Spearman, which a group-mean predictor already scores well on. The corrected metric is the mean of the per-stratum Spearman values. On it the model's probe shows some within-stratum order, but no more than the raw distances without any probe.
- **Neighbour-joining** is now also compared with trees built from class-blob inputs.
- **Numbers:** `results/qwen35_2b/SUMMARY.md`, stage 2.
