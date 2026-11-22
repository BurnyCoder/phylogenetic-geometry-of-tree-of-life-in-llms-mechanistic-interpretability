# Experiment 02 (stage 1b): triplet accuracy, Mantel, controls, conflict triplets — CONFIRMATORY

**Status:** done 2026-11-20 on the re-frozen species set.
- **Command:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases analyze,conflict
  ```
- **Numbers:** all in the auto-generated [`results/qwen35_2b/SUMMARY.md`](../../../results/qwen35_2b/SUMMARY.md) (from `results/qwen35_2b/stage1b/results.json` and `conflict/conflict.json`). This page interprets them and types no numbers by hand.

## What was tested (preregistered)
- **Representation.** Qwen3.5-2B residual stream at the last name token. Prompts are in the chat template, 8 templates per species, mean-centred per template, compared by cosine distance, at all 25 layers.
- **Primary metric (H1).** Node-balanced rooted-triplet accuracy over triplets whose three species share a stratum and share no head noun (TA_within).
  - Significance: max-T permutation over layers.
  - Point estimate: family-blocked cross-fitting.
  - CI: clade-block bootstrap.
- **Required paired comparisons** (preregistered, plus deviations 11–12):
  - random-init Qwen3.5-2B;
  - Pythia-410m step0;
  - character 3-grams;
  - the input embedding (layer 0);
  - name token count;
  - log Wikipedia popularity.
- **Secondary hypotheses** (Holm across S1–S5):
  - S1: global Mantel correlation with tree path length;
  - S2: within-order, head-noun-disjoint triplets;
  - S3: lexical-discordant triplets;
  - S4: co-occurrence (GloVe)-discordant triplets;
  - S5: divergence time beyond rank and topology.

## Results in words
1. **H1 is supported, and the effect clears the preregistered "meaningful" threshold.**
   - TA_within is significantly above chance (max-T p at the permutation floor).
   - It is above the untrained models, spelling similarity, input embeddings, token count and popularity (every paired-bootstrap CI excludes 0).
   - The cross-fitted (honest) estimate and the full-sample estimate at the modal layer agree.
   - The effect appears by layer ~6, peaks there, and stays on a plateau through the last layer.
   - **Robust** to stratum-centring (the confound route found by the review) and present on nuisance-discordant triplets (exploratory).
2. **S1 (global Mantel) and S2 (within-order) are supported.** The geometry tracks the tree globally and also *inside* orders, at family-level splits, with head nouns controlled.
3. **S3 and S4 are not supported.** On the triplets where spelling similarity (S3) or GloVe co-occurrence statistics (S4) choose the wrong pair, the model's accuracy is above 1/3 but not significantly.
   - The preregistered reading ("What would change our minds") is that the within-class signal is **not demonstrably beyond surface form and co-occurrence**.
   - A power caveat applies: these subsets are smaller and harder, and the power simulation shows effects below ~0.39 are often missed.
4. **S5 is not supported.** Controlling for topological depth, representation distance does not track TimeTree divergence time within rank bins. Consistent with H_rank: what is encoded looks like taxonomic structure, not dated phylogeny.
5. **Conflict triplets (exploratory).** Qwen picks the tree-closer partner in significantly *fewer* than half of the 36 folk-vs-phylogeny triplets:
   - blue whale is closer to whale shark than to hippopotamus;
   - Nile crocodile is closer to green iguana than to house sparrow;
   - tuna is closer to great white shark than to human;
   - horseshoe crab is closer to blue crab than to the goliath birdeater.

   Its few "phylogenetic" choices (manatee–elephant, hyrax–elephant, hedgehog–shrew) are ones a lexical or untrained baseline sometimes also gets. **The geometry follows folk / ecological categories where they conflict with phylogeny.**
6. **Per-stratum (exploratory).**
   - Within-group accuracy is high for crustaceans, sharks, echinoderms, mammals, reptiles and birds.
   - It is near chance for insects and amphibians, and *below* chance for arachnids. Spiders vs scorpions appear systematically mis-grouped, a candidate for qualitative follow-up.
7. **Untrained models are not at chance.**
   - Random-init Qwen and Pythia step0 score above 1/3 in some layers.
   - Sub-word overlap beyond head nouns (e.g. "flying fish" vs "goldfish") gives untrained networks some signal.
   - This is why beating them, not chance, is the criterion.
8. **Depth profile.** The model's advantage over the untrained and lexical baselines is largest for splits at phylum and class level and smaller at order level, consistent with coarse structure dominating.

## Interpretation (confidence-calibrated)
- **Systematic claim (supported by causal-free, correlational evidence).** Within coarse animal groups, Qwen3.5-2B's species representations are arranged in agreement with the phylogenetic tree more often than chance, untrained, lexical, token-count and popularity baselines allow. This holds down to splits inside orders.
- **What we cannot claim.** That this structure goes beyond co-occurrence statistics (S4 fails), beyond surface spelling (S3 fails), or reflects divergence time (S5 fails).
- **Where intuition and phylogeny conflict,** the representation sides with intuition: whales near sharks.
- **Overall picture.** The model has learned a *folk-taxonomic tree*: coarse classes plus a good deal of family-level structure, of the kind recoverable from text co-occurrence, rather than an evolution-faithful phylogeny.
- **Causal role.** Not yet tested (stage 3).

## Limitations specific to this experiment
- One model (2B, post-trained). Scale and other models come in stage 5.
- The conflict triplets reuse species across triplets, so the binomial test treats them as independent although they are not.
- S3/S4 subsets are selected on a baseline's errors, which makes them harder by construction for any representation correlated with that baseline.

## Revision after the final review (round 1, 2026-11-20)
Some statements above are corrected by the independent review ([review](../../reviews/2026-11-20-final-review-round1.md), [response](../../reviews/2026-11-20-final-review-round1-response.md)). The registered verdicts are unchanged. The new numbers are generated in `results/qwen35_2b/SUMMARY.md`.
- **Conflict triplets (point 5).** The untrained controls pick the tree-closer partner about as rarely as Qwen does, and exact McNemar tests show no difference. "The geometry follows folk / ecological categories" is withdrawn. The supported statement is narrower: the geometry does not follow the tree on these hand-picked, mutually dependent triplets, and neither do untrained networks.
- **S2.** S2 passes its registered test against chance, but random-init Qwen reaches the same value. On S3 and S4 the model also equals the untrained networks. Post hoc, on lexically neutral triplets (character 3-grams have no preference) the model exceeds the untrained networks.
- **Where the signal is.** Per-stratum CIs show the within-group signal is clear only in mammals and birds, which have the most decision nodes. The other strata are too small to conclude anything, and arachnids fall below chance.
- **Overall picture (point "Overall").** The phrase "folk-taxonomic tree … a good deal of family-level structure" overstates what was shown. The supported summary: modest within-group agreement with the tree beyond the registered controls, concentrated in mammals and birds, not shown to go beyond spelling or co-occurrence statistics.

## Audit note (2026-11-22)
From the [independent audit](../../reviews/2026-11-22-independent-audit.md); numbers in `results/qwen35_2b/SUMMARY.md`, section "Independent audit":
- **Reproduced.** TA_within was recomputed with independent code over every qualifying triplet, without the per-node cap of 200. The headline value and the untrained controls' values agree with the sampled ones to within sampling noise.
- **Point 7 corrected.** Random-init Qwen is *not* significantly above chance by its own max-T test over layers; only Pythia-410m step 0 is. On triplets whose names share no character 3-gram, both untrained networks are at chance (their CIs include 1/3), so shared character strings are the likely source of their signal. The "flying fish" example is not in the species set; a real pair is "housefly" and "stable fly" (both Muscidae), which differ in head noun and share no token but both contain "fly".
- **The margin over the untrained networks is fragile.** It passes the registered family-block bootstrap, also with 5 extra bootstrap seeds, but the lowest lower bound is close to zero. With orders instead of families as resampling blocks, the CIs against both untrained networks include zero. H1 remains supported as registered; the claim "beyond untrained networks" should be read as resting on the registered block level.
- **"Spelling has no preference"** is accurate: almost all of the lexically neutral triplets share no character 3-gram at all.
