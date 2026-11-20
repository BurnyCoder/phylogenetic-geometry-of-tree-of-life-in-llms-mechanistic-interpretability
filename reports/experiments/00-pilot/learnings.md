# Learnings from the pilot (Pythia-410m)

**1. The stage-1a checks are lexically confounded at the readout token.**
- At layer 0 the residual stream at the last name token is just the token embedding. For "great white shark" that token is "shark".
- Species that share a head noun are usually related: sharks, crabs, frogs.
- So kNN stratum purity (C1), order purity within strata (C2) and the tree-overlay length (C3, and even the within-stratum shuffle C3b) all beat their nulls at layer 0, without any "knowledge" beyond word identity.
- *How we found it:* the pilot's layer-0 row passed every check.
- *What we do about it:* report layer 0 as the lexical reference for all stage-1a effect sizes, and base the claim about phylogeny on H1, whose triplets are head-noun-disjoint, plus the lexical-discordant S3.

**2. C3 against a full shuffle cannot tell trees from clusters.**
- Any clustering by class shortens the overlay relative to fully shuffled leaves.
- C3b shuffles leaves only within strata, so it tests branching beyond the coarse classes. It was added as exploratory (deviation 10).

**3. The pipeline is fast.**
- Extracting 400 species × 8 templates takes about 2 minutes on Pythia-410m.
- A stage-1a analysis with 199 permutations takes about 2 minutes, and stage 1b about 2 minutes.
- So the full permutation counts (1,000) are affordable for the main run.
