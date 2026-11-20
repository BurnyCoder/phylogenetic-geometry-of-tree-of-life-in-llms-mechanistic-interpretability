# Experiment 00b: power simulation on the frozen tree (preregistration section 4.5)

**Status:** done on 2026-11-20 on the re-frozen species tree (sha256 `1f268890…`), **before** the new activations were extracted.

## Question
How large must the within-stratum triplet accuracy (TA_within) be for the H1 test to detect it reliably with 400 species and this tree?

## Method (`simulate.power_curve`, config `configs/power.yaml`)
- **Synthetic data.** Representations are simulated as a standardized class-blob component (each species at its stratum centroid plus noise, so clusters with no within-class tree) plus `signal` × a standardized Brownian-motion-on-the-tree component in 256 dimensions.
- **Signal grid:** {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}, 40 simulations each.
- **Test.** Each simulation runs the real analysis: node-balanced within-stratum, head-noun-disjoint triplets (K = 200 per node), cosine distance, and the within-stratum permutation test (199 permutations).
- **Caveat.** This is a single-layer test, so the real max-T test over 25 layers is slightly more conservative and the reported minimum detectable effect is slightly optimistic.

## Result
The table is generated in [`results/RESULTS.md`](../../../results/RESULTS.md) (section "Power simulation") from `results/power/summary.json`.
- **Detectable effect.** TA_within ≈ 0.39 is detected with 80% power, below the preregistered smallest effect of interest (0.40). So the primary test is adequately powered for any effect we would call meaningful.
- **Pure class-cluster structure (signal 0)** gives TA_within slightly *below* 1/3, i.e. chance. Class clusters alone cannot pass H1, as designed.

## Learnings
- The between-simulation SD of TA_within is about 0.02. Effects of 0.35–0.36 are therefore detected only occasionally (power 0.2–0.3), and a non-significant H1 does not rule out a small within-class signal.
