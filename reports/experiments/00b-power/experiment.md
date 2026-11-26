# Experiment 00b: power simulation on the frozen tree (preregistration section 4.5)

The synthetic experiment estimates the rejection probability of a single-layer permutation test at each signal level. The joint decision rule also requires paired control comparisons, a family confidence interval and correction for layer selection. Its power was never estimated. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#exploratory-geometry-probes-and-power) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/00b-power/experiment.md)

**Status:** done on 2026-11-20 on the re-frozen species tree (sha256 `1f268890…`), **before** the new activations were extracted.

## Question
How large must the within-stratum triplet accuracy (TA_within) be for the H1 test to detect it reliably with 400 species and this tree?

## Method (`simulate.power_curve`, config `configs/power.yaml`)
- **Synthetic data.** Representations are simulated as a standardized class-blob component (each species at its stratum centroid plus noise, so clusters with no within-class tree) plus `signal` × a standardized Brownian-motion-on-the-tree component in 256 dimensions.
- **Signal grid:** {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}, 40 simulations each.
- **Test.** Each simulation runs the real analysis: node-balanced within-stratum, head-noun-disjoint triplets (K = 200 per node), cosine distance, and the within-stratum permutation test (199 permutations).

## Reading the simulation

With only the broad-group component, within-group accuracy averaged close to chance on this tree. Increasing the simulated tree component increased detection at the higher signal settings. The generated table gives the observed rejection fraction and an exact binomial interval at every grid point. A small grid and finite repetitions limit the precision of a detectable-effect threshold. No joint-power or divergence-time claim follows from this simulation. [Power curve and intervals](../../../results/RESULTS.md#power-simulation) · [Original draws summarized by signal](../../../results/power/power.json)
