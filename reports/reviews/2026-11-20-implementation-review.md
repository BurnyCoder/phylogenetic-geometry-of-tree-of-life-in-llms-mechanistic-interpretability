# Implementation review: statistics, leakage, extraction, preregistration conformance

- **Date:** 2026-11-20
- **Reviewer role:** adversarial NeurIPS-style reviewer (Claude Code, review pass; no repository code was modified)
- **Scope:** `reports/preregistration.md` against `src/phylogeo/**` and `tests/**` at commit `b33de68`.
- **State of the run at review time:** Qwen3.5-2B extraction has **not** produced activations. `logs/bg_extract_qwen.out` ends in `OSError ... paging file is too small (os error 1455)`, and the power run (`logs/bg_power.out`) ended in `numpy._core._exceptions._ArrayMemoryError`. Only the Pythia-410m pilot has results. So every fix below can still be made **before** primary data exists, provided it is logged as a deviation.

## How the claims were checked

- `uv run pytest -q -m "not slow"`: **41 passed**, 1 deselected.
- Throwaway probes ran on the **frozen** `data/species.csv` and `data/species_tree.nwk`, using the repository's own functions. They were run from the session scratchpad, which is temporary. Each finding states what the probe computed, so it can be re-run.
- Unless noted, numbers use `sample_triplets(tree, 200, default_rng(0))`. This gives 17,825 triplets, of which 11,122 are within-stratum and head-noun-disjoint (the same counts as the pilot log).

## Summary verdict

**The core machinery is mostly correct:**
- triplet sampling, tie scoring, node-balanced mean, max-T over layers, the Mantel rank-permutation trick and Holm are implemented correctly;
- name-span extraction is exact for every prompt;
- on the real tree, the clade-block bootstrap is conservative, not anti-conservative.

**But the confirmatory layer is not ready.** There are two critical problems:
1. **The H1 decision rule can declare support without the preregistered untrained controls.** The pilot's `results.json` already reports `"supported": true` with only char-3-gram and layer-0 comparisons.
2. **Known nuisance variables pass the H1 test on their own.** A 1-D "representation" holding only the number of name tokens, or only log Wikipedia sitelinks, reaches TA_within ≈ 0.38–0.40, with within-stratum permutation p ≤ 0.015 for three different triplet-sampling seeds. None of the preregistered H1 controls targets these variables.

**Secondary hypotheses:**
- S2 passes for a purely lexical (head-noun-only) representation.
- S5 is built on ages that contain impossible zeros, rank bins that are wrong for primates and coelacanths, and a time signal that is confounded with topological depth.

**Data:** two frozen species names are mojibake, and an unregistered exclusion rule removed Hippopotamus, Dugong and others.

**Recommendation:** fix findings 1–9 (log each as a deviation) before extracting Qwen activations.

---

## Findings (most severe first)

### 1. CRITICAL: the H1 decision silently drops missing controls and adds an unregistered one
- **Where:** `src/phylogeo/analysis.py:124`, `:161`, `:165-168`; `configs/qwen35_2b.yaml` (`controls: [qwen35_2b_random, pythia410m_step0, pythia410m]`).
- **What the code does:**
  - `runs` keeps only the controls whose `last.npy` exists.
  - `req = [k for k in keys if k in runs[1:] or k in ("char3", "input_embedding")]`.
  - `beats_untrained_and_lexical = all(diffs[k][0] > 0 for k in req)`.
- **Failure scenario 1 (missing controls).** If the random-init Qwen or Pythia step0 extraction fails, which is plausible since extraction just crashed with OS error 1455, the criterion is judged only on char3 and layer 0. Those are nearly chance-level baselines (finding 2). H1 is then declared "supported" without the two untrained-model comparisons that preregistration §4.2(3) requires.
  - This already happened in the pilot: `results/pilot_pythia410m/stage1b/results.json` → `"diff_ci95_vs": {"char3": ..., "input_embedding": ...}`, `"supported": true`.
- **Failure scenario 2 (extra control).** If trained `pythia410m` activations exist, they are added to `req`. H1 would then also require Qwen to beat a *trained* model, which §4.2(3) does not list. H1 could be called falsified for a reason the preregistration does not contain.
- **Fix:**
  - Hard-code the preregistered set `{qwen35_2b_random, pythia410m_step0, char3, input_embedding}`.
  - Raise an error, or set `supported = None` with a `"missing_controls"` field, if any member is absent.
  - Report trained Pythia as descriptive only.

### 2. CRITICAL (design): popularity and name-token count alone pass the H1 machinery; no H1 control targets them
- **Where:** `src/phylogeo/analysis.py:113-118` (baselines), `:160-168` (decision); preregistration §1 ("beyond … frequency"), §4.2(3), §4.4 (no frequency baseline). C4 is only descriptive and only in stage 1a.
- **Evidence (probe).** The preregistered pipeline (`maxT_ta` with within-stratum permutation, 999 permutations) was run on 1-D distances |f_i − f_j|, with tiny symmetric jitter to break ties:

| 1-D feature | TA_within, seed 0 / 1 / 2 | within-stratum permutation p |
|---|---|---|
| number of Qwen name tokens | 0.394 / 0.401 / 0.387 | 0.003 / 0.001 / 0.008 |
| log sitelinks (popularity) | 0.379 / 0.384 / 0.382 | 0.015 / 0.008 / 0.014 |
| number of words | 0.361 / 0.373 / 0.354 | 0.11 / 0.045 / 0.19 |
| character length | 0.337 / 0.334 / 0.335 | 0.42 / 0.50 / 0.47 |

- **Interpretation:** naming conventions and popularity are phylogenetically clustered within strata. The last-token residual stream is known to encode token position and count, and plausibly frequency. So a model can reach TA_within ≈ 0.40, the preregistered smallest effect of interest, with no phylogenetic knowledge.
- **The preregistered controls do not catch this:**
  - char3 TA_within = 0.315;
  - a pure last-token-identity distance = 0.338;
  - random-init models encode token count only incidentally.
- **A second, geometric route to the same confound (probe, no phylogeny at all).** Species were simulated as stratum centroid + isotropic noise whose scale depends on popularity. Global centering + cosine, exactly as in `geometry.py`, then gives:
  - TA_within **0.372** (permutation p 0.04–0.08) when popular species are more central;
  - 0.317 with the opposite sign.
  - Within-stratum centering removes the effect (0.329 / 0.332).
  - So the preregistered distance turns norm heterogeneity linked to a clustered covariate into apparent tree signal, in either direction.
- **Fix (pre-data deviation):**
  - Add "token-count" and "log-sitelinks" 1-D baselines, or a popularity + length + token-count regression-residualized representation, to criterion 3.
  - Add a within-stratum-centered cosine variant as a sensitivity analysis.
  - Report TA_within on triplets matched for token count and popularity. For example, restrict to triplets where the tree-closer pair is not also the closest pair in |Δtokens| and |Δlog sitelinks|, in the same way S3 uses char3.

### 3. MAJOR: S2 is passed by a purely lexical representation
- **Where:** `src/phylogeo/analysis.py:109` (`in_order = np.isin(shared, ["order","family","genus"])`, not intersected with the stratum or head-noun mask) and `:175-178`.
- **Evidence (probe):**
  - A representation that is only head-noun identity + noise (no phylogeny) gives S2 TA = **0.428**, max-T p = **0.003**.
  - A pure head-noun-identity distance gives 0.452, and char3 gives 0.429.
  - In 1,196 of 4,622 S2 triplets the tree-closer pair shares a head noun (e.g. two "bears" vs a "fox").
  - Pilot Pythia S2 = 0.476, which is in the same range.
- **Fix:**
  - Define S2 as `in_order & distinct_mask(trip, heads)`. With that restriction, the head-identity distance falls to 0.333 and char3 to 0.288.
  - Or require S2 to beat char3 and head-identity baselines.
  - The preregistration text for S2 does not say "head-noun-disjoint", but §4.3 says "same TA machinery as H1", so a clarifying deviation is justified.

### 4. MAJOR: S5 uses impossible zero ages for undated nodes
- **Where:** `src/phylogeo/data/tree.py:179-207` (`date_tree`): an undated node gets `max(age of children)`, which is 0 when the children are leaves or undated nodes. Used at `src/phylogeo/analysis.py:67`.
- **Evidence:**
  - In `species_node_ages.npy`, **55 of 390** internal nodes have age 0, and 64 nodes have the same age as a child.
  - Zero-age pairs appear in every bin: genus 5 of 46, family 23 of 148, order 49 of 1,466, class 26 of 7,552, phylum 3 of 33,025.
  - A phylum-level divergence at 0 Mya is impossible.
  - The cached TimeTree responses show genuine missing data, not rate limits: 181 + 23 "No TimeTree study info available" and 79 "MRCA node not found".
- **Fix:**
  - Mark undated nodes as NaN and drop their pairs from S5. Or interpolate them, for example from the parent age and topology, and report how much was imputed.
  - Query additional representative pairs or the genus-level taxids before giving up.
  - Log as a deviation.

### 5. MAJOR: S5 (and S2, and the rank oracle) mis-bins taxa with missing OTT ranks
- **Where:** `src/phylogeo/data/species.py:258-264` (`lineage_ranks` keeps only exact rank labels); `src/phylogeo/analysis.py:36-44` (`pair_rank_level`); `src/phylogeo/stats/triplets.py:145-155`.
- **Evidence:**
  - All 8 primates (bonobo, chimpanzee, rhesus and Barbary macaques, aye-aye, ring-tailed lemur, mandrill, siamang) have an empty `order`.
  - The two coelacanths have empty class, order, family and genus.
  - Effects in S5:
    - chimpanzee–siamang (19.6 Mya) and chimpanzee–macaque (28.9 Mya) fall in the **class** bin, whose 1st percentile is 67.9 Mya;
    - the two congeneric coelacanths (25.2 Mya) fall in the **phylum** bin.
  - These are extreme, informative outliers. A model that only clusters primates would create a positive "time" correlation inside the class bin.
  - In S2, cross-family primate triplets are excluded.
  - The rank oracle treats primates as sharing no order.
- **Fix:**
  - Impute missing ranks from the nearest named ancestor at that rank in the OTT lineage, for example by using rank names case-insensitively and accepting "suborder"/"superorder" fallbacks.
  - Or compute rank sharing as "the deepest rank r such that both have the same non-empty name, or both are below the same taxon of rank ≥ r".
  - Add a unit test that chimpanzee–siamang share an order.

### 6. MAJOR: S5 cannot separate "time" from "topology", and its null is not the preregistered one
- **Where:** `src/phylogeo/analysis.py:55-86`.
- **Evidence (probe):** within rank bins, Spearman(TimeTree age, depth of the LCA node in the tree) is:
  - −0.994 in the phylum bin;
  - −0.939 in the class bin;
  - −0.637 in the order bin.
- **Consequence:** a representation that encodes only unranked topological nesting, with no time at all, produces S5 > 0. So "positive ⇒ H_time" (§3, S5) is not a valid inference.
- **The pooled statistic is dominated by coarse pairs:** the "none" and "phylum" bins hold 70,588 of 79,800 pairs (88%).
- **Null:** the preregistration says "permutation of species within rank bins". The code permutes within strata (docstring: "same as H1").
- **Fix:**
  - Use the partial Spearman of distance vs age given LCA depth or path length within bins, or bin on topological LCA depth instead of Linnaean rank.
  - Weight bins equally, or report them separately.
  - Log the change of null.

### 7. MAJOR: the point estimate and the CI target different quantities, and the CI ignores layer selection
- **Where:** `src/phylogeo/stats/triplets.py:188-215` (`crossfit_ta`), `src/phylogeo/analysis.py:151-169`.
- **What the code does:**
  - The reported point estimate is the cross-fitted TA. Each fold keeps only triplets whose three species all fall in one family half.
  - Probe: one half kept **1,593 of 11,122** triplets (14%) and **82 of 168** nodes.
  - The CI and criterion 2 instead use the bootstrap of TA on **all** triplets at the **modal** layer. That layer was chosen using the same data, and the bootstrap does not redo the choice.
- **Evidence (probe):** a single layer (so no selection effect), Brownian + class-blob signal 0.3, 15 simulations. The cross-fitted value exceeds the full-sample TA by **+0.010** (SD 0.011, SE ≈ 0.003). The cause is that the half-samples keep mainly nodes with many triplets.
- **Why it matters:** the "≥ 0.40 smallest effect of interest" decision uses the cross-fitted number, while the CI describes a different estimand. In the pilot, cross-fit 0.406 sits beside an in-sample modal-layer TA of 0.396 with CI [0.366, 0.476].
- **Fix:** choose one estimand. For example, report TA on the B half at the layer selected on A, and bootstrap that whole procedure (redo the layer choice inside each bootstrap draw). Or, at minimum, evaluate the CI on the held-out halves and state which layer the CI refers to.

### 8. MAJOR: two frozen species names are mojibake
- **Where:** `src/phylogeo/webcache.py:83-86` returns `r.text`. For `text/csv` without a charset, `requests` decodes as ISO-8859-1 (`requests.utils.get_encoding_from_headers({'content-type':'text/csv'}) == 'ISO-8859-1'`, checked).
- **Evidence:**
  - `data/species.csv` line 85 reads `GrÃ©vy's zebra`, and line 159 reads `kÄ\x81kÄ\x81pÅ\x8d` (it should be "kākāpō").
  - The same mojibake is in the QLever cache files.
  - The prompts the LLM sees contain C1 control characters, so the model gets a garbled name for 2 of 400 species. char3, GloVe (kākāpō counts as uncovered), head nouns and the polysemy check are affected too.
  - Span extraction "works" on the garbled text (0 span mismatches), so no test catches it.
- **Fix:**
  - Use `r.content.decode("utf-8")`, and repair the cached text with `text.encode("latin-1").decode("utf-8")`.
  - Re-freeze, log a deviation (still before primary data), and re-extract the pilot.
  - Add a test that no frozen name contains `Ã` or `Ä` or a C1 control character.

### 9. MAJOR: unregistered exclusion rules changed the species set
- **Where:** `src/phylogeo/data/species.py:199-203`. The code rejects titles that equal the **genus**, or that contain digits or commas.
- **What the preregistration says (§2.2.3):** reject only titles equal to the scientific name.
- **Evidence (probe on the cached candidate pool):**
  - The genus rule removed **296** candidates, including Hippopotamus (149 sitelinks), Caracal (109), Dugong (82), Addax (67), Indri (58), Jabiru and Anhinga. All of these are standard English common names.
  - The lowest-ranked selected mammal has 61 sitelinks. Hippopotamus (only member of its family in the pool) and Dugong (new family) would very likely be in the frozen set under the registered rule.
  - The digit rule removed 1 candidate and the comma rule 0.
- **Fix:** either remove the genus rule, or log it as a deviation with this list. Then re-freeze before extraction.

### 10. MINOR: the Holm family shrinks silently
- **Where:** `src/phylogeo/analysis.py:179-184`.
- **Failure:** if `species_node_ages.npy` is missing, S5 is skipped and Holm runs over 4 hypotheses instead of the preregistered 5, with no warning.
- **Fix:** fail loudly, or keep S5 with p = 1 and a flag.

### 11. MINOR: the head-noun mask stays fixed to leaves in the permutation null
- **Where:** `src/phylogeo/stats/triplets.py:179-183`. The mask is computed from the head nouns at the original leaves. Representations move under the permutation, but their head nouns do not.
- **Evidence (probe):** with a head-noun-only representation, the null SD is 0.0213 as coded vs 0.0190 when the mask moves with the species. The test therefore loses about 12% of its null precision. It is conservative, not invalid.
- **Fix:** recompute the distinct-head mask from `heads[p]` for each permutation.

### 12. MINOR: the tie rule differs from the preregistration
- **Where:** `src/phylogeo/stats/triplets.py:80-89`.
- **What differs:** a two-way tie that involves the ab pair scores 1/2. Preregistration §4.1 says "Ties count as 1/3".
- **Impact:** negligible. Only 0.17% of char3 within-stratum triplets are scored 1/2 (TA 0.3146 vs 0.3145), and the rank oracle has none.
- **Fix:** align the code or the text, and log it.

### 13. MINOR: layer 24 is not the residual stream
- **Where:** `src/phylogeo/extract.py:81`.
- **What happens:** in transformers 5.23, `capture_outputs(tie_last_hidden_states=True)` replaces `hidden_states[-1]` with the **final-norm** output (`utils/output_capturing.py:318-379`; `Qwen3_5TextModel.forward` applies `self.norm` at `modeling_qwen3_5.py:1283`). The same holds for Pythia's `final_layer_norm`.
- **Consequence:** layer 24's geometry includes RMSNorm gains, yet it enters the max-T family as if it were the residual stream.
- **Fix:** document this, or capture the pre-norm output of the last decoder layer with a hook.

### 14. MINOR: the GloVe "uninformative" fill is not uninformative
- **Where:** `src/phylogeo/baselines.py:89-90`, `src/phylogeo/geometry.py:39`.
- **What happens:**
  - After centering, the fill rows become exactly the mean, i.e. norm ≈ 2.4e-16.
  - Because of the `+1e-12` in the cosine, these rows end up at distance ≈ 1 from everything, including each other (0.99999994). Their distances to covered names range from 0.99996 to 1.00004, so float noise decides triplets that involve them.
  - Affected names: secretarybird, kākāpō (mojibake), mealworm, heikegani. This feeds the S4 discordance mask.
- **Fix:** exclude triplets with uncovered names from S4, or set their distances to NaN.

### 15. MINOR: one shared RNG stream makes results depend on unrelated settings
- **Where:** `src/phylogeo/analysis.py:98` onward. A single `rng` feeds, in order: triplet sampling, max-T, cross-fit and Mantel for every run, the noise ceiling, the bootstrap, S2–S4 and S5.
- **Failure:** the main model's cross-fit splits and bootstrap draws change when `n_perm` changes, or when a control's activations appear or disappear.
- **Fix:** `np.random.SeedSequence(cfg.seed).spawn(k)`, with one child generator per analysis and per run name.

### 16. MINOR: activations are not checked against the current species list
- **Where:** `src/phylogeo/phases.py:72-81`.
- **Failure:** if `species.csv` is re-frozen (for example after fixing finding 8 or 9), stale activations are silently misaligned with the tree leaves.
- **Fix:** assert that `names.txt` equals `df.common_name`, and store the species.csv sha256 next to the activations.

### 17. MINOR: the conflict-triplet binomial test assumes independence
- **Where:** `src/phylogeo/analysis.py:245`, `:257`.
- **Issue 1:** triplets share species (human ×5, African bush elephant ×4, red kangaroo ×3, cat ×3, Atlantic cod ×3), so binomial p-values are anti-conservative. The test is exploratory, but this should be stated, or a permutation or cluster bootstrap over anchors used.
- **Issue 2:** ties are counted as folk choices.
- **Verified:** all 36 triplets are resolved, and in all 36 the tree-closer partner equals the preregistered phylogenetic partner.

### 18. MINOR: rate-limit responses would be cached forever
- **Where:** `src/phylogeo/webcache.py:62`. `400 <= status < 500` with `accept_error_json` also caches **429** responses permanently.
- **Status:** no 429 is in the current cache (checked), but a rerun could poison it.
- **Fix:** exclude 408 and 429.

### 19. MINOR: TA_within mostly measures deeper within-stratum nodes
- **Where:** the design (preregistration §4.1), as implemented in `src/phylogeo/stats/triplets.py:41-77`, `:92-119`.
- **Evidence:**
  - Only 168 nodes have a qualifying triplet.
  - 33 of them have exactly 1 triplet and 54 have fewer than 5, each still weighted equally. Their population counts are small too (median 1), so this is not a sampling artifact.
  - The head-noun filter removes all triplets from 78 nodes, mostly genus- and family-level nodes such as the two bears.
- **Consequence:** TA_within is informative mainly about deeper within-stratum nodes. This should be stated when interpreting H1.

### 20. MINOR: the stage-1a reading rule is looser than the preregistration
- **Where:** `src/phylogeo/viz/pca.py:128-137`.
- **Issues:**
  - C2 "passes" if **any** of the three strata passes (3 uncorrected tests).
  - The preregistered "confound-driven" reading (top PCs correlate more with popularity or token count than with any stratum) is not implemented.

### 21. MINOR: preregistered baselines and variants are not computed in stage 1b
- **Where:** `src/phylogeo/analysis.py:113-118`. §4.4 says "every metric is computed for all of them".
- **Missing:**
  - the class-blob simulation, which also is not "matched to the real per-class spread" (`simulate.py:33-37` uses a fixed spread of 1);
  - the Brownian-tree simulation;
  - the raw and z-scored cosine variants (§2.5);
  - the mean-over-name-tokens secondary readout.
- **Also:** the power simulation uses 256-d (`phases.py:250`), not the 2048-d stated in §4.4.

### 22. MINOR: unpinned or unrecorded provenance
- `phases.py:127` loads the tokenizer without the pinned revision.
- `configs/pythia410m.yaml` has no revision.
- `FREEZE.json` does not record the Open Tree synthesis version or the QLever/Wikidata snapshot date.
- `batch_size: 32` in `RunConfig` is unused, which contradicts deviation 7 in wording only.
- GPU kernel determinism (Gated DeltaNet / SDPA in bf16) is assumed ("exact and reproducible", deviation 7) but not tested.

### 23. MINOR: test gaps
- No tests cover:
  - the H1 decision logic (missing or extra controls);
  - `holm`;
  - `s5_rank_vs_time`;
  - what the bootstrap weights mean (for example, that drawing a family twice is equivalent to duplicating its leaves);
  - text encoding of the frozen names.
- The existing type-I tests are weak: 20 simulations with a "≤ 0.2" threshold.

### 24. PROCESS: order of operations and crashed jobs
- Preregistration §4.5 says the power simulation runs "before any real activation is viewed".
- Pilot activations have been viewed (pilot results exist, deviation 10), but the power run crashed with a MemoryError. Running it in parallel with extraction exhausted the paging file.
- Fix: run the jobs one after another.

---

## Checks that passed (negative results, reported for completeness)

| Check | Result |
|---|---|
| Triplet sampling is uniform per node; polytomies | Child chosen ∝ C(\|C_i\|,2)(n_v−\|C_i\|), then a uniform pair and a uniform outgroup, which is uniform over resolved triplets. Only resolved triplets are produced. The frozen tree has 9 three-way polytomies and 381 binary nodes. |
| Name-span location inside the chat template (Qwen) and with BOS (Pythia) | Covers all 400×8 species prompts and 79×8 conflict prompts. **0** decoded-span mismatches, and the last token never extends past the name. The chat text is `<\|im_start\|>user\n…<\|im_end\|>\n` with no system prompt. The BOS shift is correct. |
| Last-token (subword) leakage in head-noun-disjoint triplets | Compound heads such as goldfish/catfish and housefly/dragonfly share a last token. But among TA_within triplets, the tree-closer pair shares the last token in 69 cases and a wrong pair in 70. A pure last-token-identity distance gives TA_within 0.3375 ≈ chance. **No bias** from this route. |
| Per-template centering | Equals global centering of the template means (the two operations commute). It carries no tree information, so it does not leak into cross-fitting. |
| Clade-block bootstrap calibration on the real tree | Null (random vectors, 150 simulations): bootstrap SD 0.0246 vs across-simulation SD 0.0189; P(lower bound > 1/3) = 0.000; P(CI excludes 1/3) = 0.020. Signal: coverage of the true value 0.98. Conservative. |
| Max-T over layers | One permutation per draw for all layers; p = (1 + #{max_perm ≥ max_obs}) / (B + 1). Correct. |
| Mantel rank trick | Permuting rows and columns maps the multiset of upper-triangle values onto itself, so ranks permute along with the values. r matches scikit-bio (tests). |
| Holm | Monotone step-down; correct. |
| Random-init Qwen | `AutoModelForCausalLM.from_config(Qwen3_5Config)` builds the text-only `Qwen3_5ForCausalLM`: 1.882B parameters, 24 layers, no vision tower (checked on the meta device). |
| Wikidata-stratum prefilter | After common-name filtering, every prefilter threshold is below the lowest selected sitelinks in the same stratum, and no selected species has a Wikidata stratum guess different from its OTT stratum. Very likely harmless. |
| Stratum monophyly in the frozen tree | All strata are clades except reptiles (paraphyletic with respect to birds, as expected) and "other". |

## Preregistration mismatches (code vs registered text)

1. **H1 criterion 3 control set** (§4.2): missing controls are dropped silently, and trained Pythia-410m is added (finding 1).
2. **"Recompute TA_within at the cross-fitted layer"** (§4.2): the code uses the modal layer on all triplets. This is not cross-fitted, the layer choice is not repeated inside the bootstrap, and the estimand differs from the point estimate (finding 7).
3. **Tie rule** (§4.1): a two-way tie involving ab scores 1/2, not 1/3 (finding 12).
4. **S5 null** (§4.3): the code permutes within strata, not within rank bins. S5 also has no layer max-T and is evaluated at the TA modal layer (findings 4–6).
5. **Holm family of 5** (§3, §4.3): it becomes 4 silently when ages are missing (finding 10).
6. **S2 subset** (§3): it includes triplets split *below* genus (all three species share a genus), triplets with shared head nouns, and the "other" stratum (2 triplets). Cross-family primate triplets are excluded because of the missing order rank (findings 3 and 5).
7. **Species rule** (§2.2.3): genus-equal, digit and comma exclusions are unregistered, and 2 names are not the Wikipedia title (mojibake) (findings 8 and 9).
8. **Rank oracle** (§4.4: "depth of the deepest shared Linnaean rank"): the code uses 5 − (number of shared ranks), which differs when ranks are missing (10 species).
9. **§4.4 "every metric is computed for all baselines"**: the class-blob and Brownian simulations are absent from stage 1b. The class blob is not matched to the per-class spread. `tree_oracle` is an unregistered extra, which is harmless.
10. **§2.5 sensitivity variants and the mean-token readout**: implemented in `layer_distances` but never run by `run_stage1b`.
11. **§2.5 "residual stream hidden_states[ℓ], ℓ = 0…24"**: ℓ = 24 is the final-norm output (finding 13).
12. **§3 H1a reading rule**: C2 passes on "any stratum", and the "confound-driven" reading is not implemented (finding 20).
13. **§4.3 S1 "value reported at the cross-fitted layer"**: the code reports S1 at the TA-modal layer, not at a Mantel-specific cross-fitted layer. This is defensible but should be stated.
14. **§1 question "beyond … frequency"**: no frequency or popularity control exists in stage 1b (finding 2).
15. **§4.5 power before viewing real activations**: the power run has not completed, while pilot activations have been viewed (finding 24).
16. **§4.4 Brownian simulation "in 2048-d"**: the power simulation uses 256-d.

## Suggested order of fixes (all can be done before Qwen extraction)

1. Encoding fix → remove or log the genus rule → re-freeze → log the deviations (findings 8, 9, 16).
2. Make the H1 decision use exactly the preregistered controls and fail loudly when one is missing (finding 1).
3. Add token-count and popularity controls, plus a within-stratum-centered sensitivity variant, as a pre-data deviation (finding 2).
4. Restrict S2 to head-noun-disjoint triplets. Fix the S5 ages, rank imputation, the topology confound and the null (findings 3–6).
5. Unify the estimand of the point estimate and the CI (finding 7). Use independent RNG streams (finding 15).
6. Re-run the power simulation alone, then extraction, then the controls, one at a time.
