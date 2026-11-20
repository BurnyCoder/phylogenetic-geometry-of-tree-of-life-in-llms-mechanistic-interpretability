# Experiment 06: the habitat gradient (preregistration Addendum A), tested on held-out species

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
The axis is PC1 of the layer-21 species vectors of the 400 primary species, oriented so birds are positive and ray-finned fish negative. Held-out activations are centred with the primary set's per-template means and projected onto the axis. Tests are one-sided, Holm-corrected across the evaluable tests.

## Results in words
- **Aquatic end: confirmed.**
  - Held-out marine mammals sit far toward the aquatic end compared with ground-dwelling mammals (M1).
  - Among held-out birds, the more a species forages in water, the lower its axis score (B1).
  - Both survive Holm correction.
- **Aerial end: not confirmed.**
  - Aerial-foraging birds do not score higher (B2). The correlation even has the opposite sign, though only 9 held-out birds forage aerially at all.
  - The bat test (M2) is not evaluable: there are no bats in the held-out set (preregistered rule: fewer than 5 → descriptive only).

## Interpretation
The dominant top-PC direction of the late-layer species geometry encodes an **aquatic ↔ terrestrial habitat axis** that cuts across phylogeny. Whales and seals move toward fish; waterbirds move away from other birds. This confirms out of sample that the representation mixes ecology with taxonomy. It also explains why the top PCs do not look like a nested tree (stage 1a), and why where ecology and phylogeny conflict, the geometry follows ecology (stage-1b conflict triplets).

The "air" part of the stage-1a ordering (birds and insects at one end) is not supported within birds. That end may reflect class identity (birds and insects as large groups) rather than flight per se.

## Limitations
- Only 6 marine mammals.
- No bats.
- Mammal foraging strata are coarse categorical codes.
- The axis was defined at one layer chosen during exploration (layer 21). That choice is part of the registered addendum, but it was informed by the stage-1a plots.

## Revision after the final review (round 1, 2026-11-20)
The interpretation above overstates the result in three places ([review](../../reviews/2026-11-20-final-review-round1.md), S3):
- **Variance share.** The axis is the *top* late-layer component, but it holds only a few percent of the variance. "Dominant" is replaced by "top component".
- **Name cues.** All 6 marine mammals have aquatic cues in their names (whale, seal, porpoise, sea otter), and water foraging in birds is phylogenetically clustered (ducks, grebes, loons). Name cues were not controlled.
- **Link to the conflict triplets is withdrawn.** The untrained networks behave the same on those triplets (final review M4).

The registered reading stands: the geometry mixes ecology with taxonomy.
