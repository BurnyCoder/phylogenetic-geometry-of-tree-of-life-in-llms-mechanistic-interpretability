# Experiment 05 (stage 4): ablations

**Status:** done 2026-11-20 (exploratory).
- **Commands:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases stage4
  ```
  Extra extractions, run first:
  - `--phases extract --name-sets species_sci` (scientific binomials);
  - `configs/qwen35_2b_raw.yaml` (no chat template).
- **Numbers:** auto-generated in [`results/qwen35_2b/SUMMARY.md`](../../../results/qwen35_2b/SUMMARY.md), section "Stage 4".

## What changes, one at a time
- Readout: last name token vs mean over name tokens.
- Distance normalization: per-template centering (main) vs raw cosine, z-scored dimensions, stratum-centered.
- Prompt format: chat template vs raw text.
- Names: common names vs scientific binomials. Binomial triplets are split above genus, because congeners share the genus token.
- Species set: polysemous names included (414 species, combined Open Tree tree).
- Probe:
  - dimension k ∈ {2, 3, 5, 10, 20};
  - metric angular / Euclidean / Poincaré at k ∈ {2, 5, 10};
  - within-group held-out evaluation, compared with the shuffled-tree probe.
- Geodesic: the kNN k used for the graph.

## Results in words
1. **Robust to the incidental choices.**
   - z-scoring dimensions, stratum-centring, dropping the chat template and including polysemous names all leave TA_within essentially unchanged.
   - The effect is not an artifact of the normalization, the chat format or the exclusion rule.
2. **Readout matters.**
   - Raw cosine without centering is weaker: the shared anisotropic direction dominates.
   - Averaging over all name tokens is weaker than taking the last name token. Earlier tokens of multi-word names ("great", "white") carry little species identity.
3. **Common names, not Latin binomials, carry the structure.**
   - With binomial prompts, TA on genus-split triplets is at chance at the main layer, and only modestly above it at the best layer.
   - Common names on the *same* triplets score clearly higher.
   - The model's taxonomic geometry is tied to vernacular names, as expected from text co-occurrence.
4. **The probe captures within-class structure, not only class membership.**
   - The within-group held-out Spearman of the k = 10 probe is clearly above the shuffled-tree probe's.
   - This answers the caveat raised in stage 2.
5. **Low dimensionality.** Held-out probe correlation is already substantial at k = 2 and saturates around k = 10.
6. **Hyperbolic is not better than flat.** The Poincaré probe never beats the Euclidean probe at equal k. The preregistered prediction "Poincaré beats Euclidean at d ≤ 10" (H_geom) is **not supported**.
7. **Geodesic distance depends on k.**
   - At the smallest connected k the sparse graph hurts.
   - At larger k the global Mantel correlation rises above angular, but TA_within does not.
   - Geodesics change the scale of large distances, not local tree order.

## Learnings
- Report the readout and normalization explicitly. They change the headline number by more than most "method" choices.
- Using binomials is not a neutral substitution for common names. LLM knowledge is indexed by the names people write.
- A hyperbolic probe is not automatically better for tree targets. With linear maps into low dimensions, the flat metrics did as well or better here.

## Revision after the final review (round 1, 2026-11-20)
Corrections from the [review](../../reviews/2026-11-20-final-review-round1.md):
- **Point 4 is withdrawn.** "The probe captures within-class structure" relied on the pooled within-group Spearman, which a group-mean predictor already saturates (M2). With the per-stratum metric and a class-blob control, the probe's within-stratum order is real but no larger than that of the raw distances.
- **Point 3 is softened.** On genus-split triplets that are also head-noun-disjoint, binomials carry much weaker structure that appears only in late layers. They are not structureless. The common-name comparator on those triplets is lower than the figure quoted above.
- **Point 6 is made precise.** The Poincaré probe is never better than the best flat (Euclidean) probe. At k = 2 it does beat the angular probe.
