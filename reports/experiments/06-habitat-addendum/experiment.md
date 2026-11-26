# Experiment 06: the habitat gradient (preregistration Addendum A), tested on held-out species

The marine-mammal and water-foraging bird predictions meet their four-slot Holm criteria on held-out species. Aerial mammals were absent and the aerial-bird prediction did not meet its directional criterion. Both aquatic associations also survive the within-order permutation sensitivity. The marine species belong to two orders and may share lexical cues. [Independent analysis](../../reviews/2026-11-26-scientific-audit.md#habitat-on-held-out-species) · [Dated original experiment record](https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/cc36b9134981497d6006359183261381c25177ac/reports/experiments/06-habitat-addendum/experiment.md)

**Status:** done 2026-11-20.
- **Registration:** the hypothesis and analysis were registered in commit `31d0e99`, **before** the held-out species were selected or extracted.
- **Commands:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases extract --name-sets heldout_eco
  ```
  then `--phases eco`.
- **Numbers:** auto-generated in [`results/qwen35_2b/SUMMARY.md`](../../../results/qwen35_2b/SUMMARY.md), section "Addendum A".

## Why
In stage 1a the dominant late-layer PCA axis ordered strata from aquatic groups (sharks, marine invertebrates, fish) through tetrapods to flyers (insects, birds). That was a post-hoc observation, so it was turned into a directional hypothesis and tested on **new species**.

## Data
- 79 held-out mammals and 80 held-out birds from the same candidate pool and rules as the primary set, none in it.
- Foraging strata from EltonTraits 1.0 ([Wilman et al. 2014](https://esapubs.org/archive/ecol/E095/178/); files MD5-verified against the published checksums):
  - mammals: a `ForStrat-Value` code (M marine, G ground, S scansorial, Ar arboreal, A aerial);
  - birds: percentages of foraging below or around the water surface, and of aerial foraging.
- The held-out mammals include 6 marine species (sea otter, monk seal, sei whale, harbour porpoise, bearded seal, minke whale) and **no bats**.

## Method
The axis is PC1 of the layer-21 species vectors of the 400 primary species, oriented so birds are positive and ray-finned fish negative. Held-out activations are centred with the primary set's per-template means and projected onto the axis. The original tests were one-sided and used the available members of the family. The corrected report accounts for all four registered tests, assigning no result to aerial mammals.
