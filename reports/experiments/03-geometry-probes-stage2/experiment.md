# Experiment 03 (stage 2): manifold / tree geometry and the trained flat-subspace probe

The global probe correlations are compatible with coarse categories. Historical fits used a target-scale denominator that included held-out species and a layer chosen from the full panel. The raw within-stratum comparator uses the identical family folds. A normalized Robinson–Foulds distance cannot by itself be read as a fraction of true splits recovered. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#exploratory-geometry-probes-and-power) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/03-geometry-probes-stage2/experiment.md)

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
