# Preregistration: Phylogenetic geometry of animal species in an LLM

- **Author:** Libor "Burny" Burian
- **Date registered:** 2026-11-20. Committed to the public GitHub repository *before* any activation has been extracted or viewed. The commit hash is the timestamp.
- **Status:** No data on model representations exists at registration time.
- **Changing this document:** any change after registration goes in the **Deviations** section at the end, with a date and a reason.

---

## 1. Domain and question

**Domain.** The mechanistic interpretability of how large language models represent structured world knowledge, here the tree of life.

**Question.** Do the internal representations of animal species names in an LLM have **phylogenetic geometry**? Concretely:
- Does the distance between two species' representations reflect how closely related they are on the tree of life?
- Does this hold *beyond* folk categories (bird / mammal / fish / bug), surface form, frequency and co-occurrence?
- What shape does this structure have: tree, clusters, gradient, or manifold?

**Motivation.**
- Goodfire found a curved phylogenetic manifold in the DNA model Evo 2 ([post](https://www.goodfire.com/research/phylogeny-manifold)).
- Goodfire's neural-geometry series found circles and other manifolds for concepts such as days of the week ([index](https://www.goodfire.com/research/neural-geometry)).
- No peer-reviewed study tests a text LLM against the phylogeny ([literature review](literature-review.md)).

---

## 2. Materials

### 2.1 Model
- **Primary:** [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), the standard post-trained release. Revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, bf16, run entirely on one RTX 5070 Laptop GPU (8 GB).
- **Untrained controls:**
  - Qwen3.5-2B with randomly initialized weights (`from_config`, seed 0);
  - Pythia-410m `step0` ([EleutherAI](https://huggingface.co/EleutherAI/pythia-410m)).
- **Pilot / dev model:** Pythia-410m (final checkpoint). It is used only to debug the pipeline. Its results are reported, but they are not used to change any hypothesis or analysis.

### 2.2 Species set (frozen and sha256-hashed before activations are extracted)
Rule, implemented in `src/phylogeo/data/species.py`:

1. **Strata.** Thirteen coarse strata, each defined by a taxon:
   - mammals (Mammalia), birds (Aves), reptiles (Reptilia, excluding birds), amphibians (Amphibia), ray-finned fish (Actinopterygii), cartilaginous fish (Chondrichthyes);
   - insects (Insecta), arachnids (Arachnida), crustaceans (Crustacea);
   - molluscs (Mollusca), cnidarians (Cnidaria), echinoderms (Echinodermata);
   - other (remaining animal phyla and groups, e.g. Annelida, Porifera, Myriapoda, jawless fish, lobe-finned fish).
   - A species is assigned to the first matching stratum in this order.
2. **Candidates.** Wikidata items with taxon rank = species, in Animalia, with an English Wikipedia article and an NCBI taxid (P685).
3. **Common name.**
   - The common name is the English Wikipedia article title, lowercased with any parenthetical removed.
   - A candidate is kept only if the title is a common name, i.e. it differs from the scientific name (P225).
4. **Ranking.** Within each stratum, candidates are ranked by their number of Wikipedia sitelinks, a standard notability proxy.
5. **Family cap.** Candidates are taken in rank order with at most **3 species per family**, until the stratum quota is met.
   - Quotas, targeting about 400 in total: mammals 90, birds 70, reptiles 30, amphibians 20, ray-finned fish 45, cartilaginous fish 15, insects 45, arachnids 15, crustaceans 15, molluscs 20, cnidarians 10, echinoderms 10, other 15.
6. **Open Tree match.** Every species must match an Open Tree Taxonomy id through TNRS and appear in the Open Tree induced subtree. Species that fail are dropped and logged.
7. **Polysemy flag.**
   - A name is flagged when:
     - it is a single word whose first WordNet noun sense is not `noun.animal`, or
     - it is on this preregistered list of known non-animal senses: jaguar, cougar, wolverine, puma, impala, mustang, falcon, hornet, viper, cobra, python, lynx, panther, raven, eagle, swift, kite, crane, seal, bass, sole, turkey, mole, bat, cardinal, robin, buffalo.
   - Flagged names are **excluded from the primary set** and analyzed separately in a sensitivity analysis.
8. **Duplicates.** Duplicate common names are both dropped.

The final N and the class composition are recorded in `data/species.csv` with its sha256.

### 2.3 Conflict triplets (preregistered list)
Each triplet is (anchor, phylogenetic partner, folk/lexical partner). The *phylogenetic partner* is the one predicted by the folk-taxonomy-violating fact. The **correct answer is computed from the reference tree, not from this table.** A triplet the tree does not resolve is excluded.

| # | Anchor | Phylogenetic partner | Folk/lexical partner | Conflict type |
|---|---|---|---|---|
| 1 | blue whale | hippopotamus | whale shark | ecology + lexical |
| 2 | nile crocodile | house sparrow | green iguana | folk "reptile" |
| 3 | atlantic bluefin tuna | human | great white shark | folk "fish" |
| 4 | common starfish | human | common octopus | folk "sea creature" + lexical |
| 5 | atlantic horseshoe crab | goliath birdeater | blue crab | lexical "crab" |
| 6 | little brown bat | cattle | house mouse | folk "flying mouse" |
| 7 | rock hyrax | african bush elephant | guinea pig | folk "rodent-like" |
| 8 | thylacine | red kangaroo | gray wolf | convergence |
| 9 | european eel | human | sea lamprey | folk "eel-like" |
| 10 | western honey bee | american lobster | goliath birdeater | folk "bugs" |
| 11 | sugar glider | red kangaroo | southern flying squirrel | convergence |
| 12 | aardvark | african bush elephant | giant anteater | convergence |
| 13 | chinese pangolin | cat | giant anteater | convergence |
| 14 | european hedgehog | common shrew | north american porcupine | convergence (spines) |
| 15 | west indian manatee | african bush elephant | walrus | ecology |
| 16 | coelacanth | human | atlantic bluefin tuna | folk "fish" |
| 17 | australian lungfish | human | atlantic salmon | folk "fish" |
| 18 | komodo dragon | king cobra | nile crocodile | folk "lizard-like" |
| 19 | acorn barnacle | edible crab | common limpet | folk "shell" |
| 20 | common woodlouse | american lobster | seven-spot ladybird | folk "bug" |
| 21 | elephant shrew | african bush elephant | common shrew | lexical "shrew" |
| 22 | red panda | raccoon | giant panda | lexical "panda" |
| 23 | koala | red kangaroo | brown bear | folk "koala bear" |
| 24 | killer whale | bottlenose dolphin | blue whale | lexical "whale" |
| 25 | indian flying fox | little brown bat | red fox | lexical "fox" |
| 26 | california sea lion | harbor seal | lion | lexical "lion" |
| 27 | common starfish | purple sea urchin | atlantic cod | lexical "fish" |
| 28 | moon jellyfish | sea anemone | atlantic cod | lexical "fish" |
| 29 | silverfish | western honey bee | goldfish | lexical "fish" |
| 30 | guinea pig | house mouse | pig | lexical "pig" |
| 31 | seahorse | atlantic cod | horse | lexical "horse" |
| 32 | electric eel | channel catfish | european eel | lexical "eel" |
| 33 | spotted hyena | cat | dog | folk "dog-like" |
| 34 | fossa | meerkat | cat | folk "cat-like" |
| 35 | pronghorn | giraffe | impala | folk "antelope" |
| 36 | bottlenose dolphin | cattle | great white shark | ecology |

The names are resolved through Open Tree TNRS to species, or to the type species for genus-level names such as seahorse, using the same pipeline. Their representations are extracted with the same templates. They form a separate set and are **not** part of the 400-species primary analysis.

### 2.4 Reference trees
- **Primary: topology.** The Open Tree of Life synthetic tree, induced subtree over the species' OTT ids ([API v3](https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3); data CC0). Polytomies are kept as polytomies.
- **Ranks.** Open Tree Taxonomy lineage gives phylum, class, order, family and genus for the depth profile.
- **Dated tree (secondary, for S5 only).** Each internal node of the Open Tree topology is dated with a TimeTree 5 MRCA age ([API](https://timetree.temple.edu/api)), queried with one representative species from each of two child subtrees.
  - Ages are clamped so that a parent is never younger than its child.
  - TimeTree data is not redistributed: it is cached locally only, and the refetch code is published.

### 2.5 Prompts and readout
- **Templates.** Eight templates. Each mentions the species once, uses the article "the", and contains no class words:
  1. `The {name} is an animal.`
  2. `Here is a fact about the {name}.`
  3. `Scientists studied the {name} in the wild.`
  4. `This animal is called the {name}.`
  5. `Tell me about the {name}.`
  6. `A photo of the {name}.`
  7. `My favorite animal is the {name}.`
  8. `We read a book about the {name}.`
- **Formatting for Qwen3.5-2B.** Each template is the content of a single user turn in the official chat template (`apply_chat_template`, no generation prompt).
- **Formatting for Pythia.** Raw text with BOS prepended.
- **Name span.** Located with tokenizer offset mappings.
- **Readout.** The residual stream (`hidden_states[ℓ]`, ℓ = 0…24, where 0 is the embedding output) at the **last token of the species name**. The mean over the name's tokens is a secondary readout.
- **Species vector, per layer:**
  1. For each template, subtract the mean over species.
  2. Average over the 8 templates.
- **Distance.** Cosine distance between species vectors. Sensitivity variants: raw (uncentered) cosine, and per-dimension z-scored cosine. Whitening is never used for a primary analysis.

---

## 3. Hypotheses

### Stage 1a (run first): "Do they look like a tree?" (exploratory, visual)

**H1a.** In the top-3 PCA directions of some layer, species form a branching tree shape:
- Major clades occupy distinct arms that radiate from a common center, rather than separate islands.
- Sub-clades split off inside their parent arm (nested branching).
- The reference tree, drawn in PC space, is compact.

**Alternatives to distinguish:**
- (a) an isotropic blob;
- (b) separated class clusters with no internal branching;
- (c) a continuous ecological or size gradient;
- (d) a curve or circle;
- (e) structure driven by frequency or token count.

**Pre-specified checks.** Run per layer, on the top 3 PCs of the species vectors. Nulls: 1,000 label permutations, plus the random-init Qwen model.

| Check | What it measures | Null |
|---|---|---|
| C1 | kNN (k=5) stratum purity and silhouette | Label permutation; random-init Qwen |
| C2 | Order-level kNN purity in a PCA computed *within* mammals, birds, and insects separately | Within-stratum permutation |
| C3 | Tree-overlay length: sum of Euclidean edge lengths of the reference tree, with each internal node at the centroid of its descendant leaves in PC1–3 | Shuffled leaf labels |
| C4 | Spearman correlation of PC1–3 with stratum indicators, log Wikipedia sitelinks (popularity), token count, and name length | (descriptive) |

**Reading the result:**
- **"Looks like a tree":** C1, C2 and C3 all beat their nulls (p < 0.05, uncorrected; this stage is exploratory) in the same layer, and the plot shows arms rather than islands.
- **"Clusters, not a tree":** C1 passes, C2 and C3 fail.
- **"Confound-driven":** the top PCs correlate more with popularity or token count than with any stratum indicator.
- **"No structure":** C1 fails in every layer.

**Anti-bias procedure.**
- The shape is first described from *uncolored* plots, then from colored plots with the tree overlay. Both descriptions are recorded.
- Calibration plots use the same code:
  - days of the week and months, where a circle is expected (Engels et al. 2024);
  - random-init Qwen;
  - a synthetic Brownian-motion-on-the-reference-tree point cloud.
- A negative H1a does **not** falsify phylogenetic geometry in higher dimensions; Stage 1b tests that.

### Primary hypothesis (confirmatory)

**H1 (within-class phylogenetic geometry).** In Qwen3.5-2B, species representations reflect the tree *within* coarse classes. Triplets are therefore resolved in agreement with the reference tree above chance.

### Confirmatory secondary hypotheses (Holm-corrected, family of 5)
- **S1 (global, Goodfire-comparable).** The Mantel Spearman correlation between representation distance and reference-tree path length (number of edges) is greater than 0.
- **S2 (fine depth).** Triplet accuracy for triplets within one taxonomic order (split at family or genus level) is greater than 1/3.
- **S3 (beyond surface form).** Triplet accuracy on *lexical-discordant* within-stratum triplets is greater than 1/3. These are triplets where character-3-gram distance picks the wrong pair.
- **S4 (beyond co-occurrence).** Triplet accuracy on *co-occurrence-discordant* within-stratum triplets is greater than 1/3. These are triplets where static GloVe distance (`glove-wiki-gigaword-300`, averaged over words) picks the wrong pair.
- **S5 (rank vs time, two-sided).** Within bins of equal shared Linnaean rank, the Spearman correlation between representation distance and TimeTree divergence time differs from 0.
  - Positive ⇒ the geometry tracks dated phylogeny (H_time).
  - ≈ 0 ⇒ it tracks rank only (H_rank).

### Causal hypothesis (confirmatory, stage 3 — renumbered from stage 6 on 2026-11-20, see Deviations; separate Holm family of 2)
- **C-abl.** Mean-ablating the learned k=10 phylogeny subspace (stage 2 flat probe) at the selected layer lowers the log-probability of the correct class word in `The {name} is a kind of ___` more than ablating a variance-matched random subspace drawn inside the span of the species activations. Measured on held-out species.
- **C-swap.** Swapping the subspace coordinates of species A into species B switches the predicted class word to A's class more often than a random-subspace swap.

### Exploratory (BH-FDR at q = 0.05 across all exploratory tests)
- H1a checks.
- Geometry and tree-likeness:
  - geodesic vs angular distance correlation, compared with the class-blob null;
  - Gromov δ_rel vs a spectrum-matched Gaussian and vs non-hierarchical concept sets;
  - neighbor-joining tree vs the reference (normalized RF and triplet distance) vs random trees.
- Probes:
  - flat-probe held-out-family correlation and selectivity;
  - Poincaré vs Euclidean probe.
- Conflict triplets (binomial test against 0.5, and the share of triplets that follow the folk partner) and behavioral triplet judgments.
- Model scale and training-dynamics curves.
- Days/months positive controls.
- SAE feature analysis.
- All ablations and sensitivity analyses.

---

## 4. Primary metric and analysis

### 4.1 Rooted-triplet accuracy (TA)
- **Which triplets count.** For species a, b, c, the triplet is *resolved* by the reference tree when MRCA(a,b) is a strict descendant of MRCA(a,b,c). Triplets inside an unresolved polytomy are excluded.
- **Correctness.** A resolved triplet is *correct* when d(a,b) < min(d(a,c), d(b,c)). Chance is 1/3. Ties count as 1/3.
- **Node-balanced sampling.**
  - The decision node is v = MRCA(a,b,c).
  - For every internal node v, sample up to **K = 200** resolved triplets with a and b in one child subtree of v and c in another (seed 0).
  - TA is the mean, across nodes, of each node's mean correctness. Every node has equal weight, so large clades cannot dominate.
- **TA_within (primary statistic).**
  - Only triplets where all three species are in the same stratum (excluding stratum "other") and all three head nouns differ.
  - The head noun is the last word of the common name.
  - Nodes with no qualifying triplet are skipped.

### 4.2 Test of H1
- **Statistic:** TA_within at every layer ℓ = 0…24.
- **p-value (max-T).** Permute species↔leaf assignments *within strata*, 1,000 times, applying the same permutation to all layers. Record the maximum TA_within over layers for each permutation. p = (1 + #{max_perm ≥ max_obs}) / 1001.
- **Point estimate (family-blocked 2-fold cross-fitting).**
  - Split families at random (seed 0) into halves A and B.
  - Pick the layer with the highest TA_within on triplets made of species from A; evaluate that layer on triplets made of species from B. Repeat with A and B swapped, and average.
  - Repeat over 20 random family splits and report the mean.
- **95% CI (clade-block bootstrap).**
  - Resample *families* with replacement, 1,000 times.
  - Recompute TA_within at the cross-fitted layer.
  - Drop any triplet that contains two copies of the same species.
  - Report the percentile interval.
- **H1 is supported when all of these hold:**
  1. max-T p < 0.05;
  2. the lower bound of the 95% CI is above 1/3;
  3. the paired bootstrap difference TA_within(Qwen3.5-2B) − TA_within(control) has its 95% CI above 0 for each of: random-init Qwen3.5-2B, Pythia-410m step0, the character-3-gram baseline, and the Qwen input-embedding baseline (layer 0).
- **Smallest effect of interest:** TA_within ≥ 0.40. A significant effect below 0.40 is reported as "statistically detectable but small".
- **H1 is falsified** if criterion 1 or 2 fails, or if TA_within does not exceed the untrained and lexical baselines.

### 4.3 Secondary analyses
- **S1.**
  - Statistic: Spearman Mantel correlation between cosine distance and tree path length.
  - p-value: max-T over layers with 1,000 full label permutations.
  - Value reported: the cross-fitted layer.
  - Kendall τ-b is also reported because of the many ties.
- **S2–S4.** Same TA machinery as H1, applied to the stated triplet subsets.
- **S5.**
  - Pairs are binned by the deepest Linnaean rank they share.
  - Within each bin, compute Spearman(cosine distance, TimeTree age).
  - Pool the bins with weights equal to pair counts.
  - Null: permutation of species within rank bins.
- **Multiplicity.**
  - Holm across S1–S5 at α = 0.05.
  - Layers are handled by max-T.
  - Exploratory tests use BH at q = 0.05.

### 4.4 Baselines and controls (every metric is computed for all of them)

| Name | What it is | Role |
|---|---|---|
| Class oracle | distance 0 within stratum, 1 between | Upper bound for "folk categories only" |
| Rank oracle | distance = depth of the deepest shared Linnaean rank | Linnaean upper bound |
| Char-3-gram | Jaccard distance of character 3-gram sets of the names | Lexical baseline |
| Input embedding | Qwen layer-0 vectors (`hidden_states[0]`) | Static lexical baseline |
| GloVe | `glove-wiki-gigaword-300`, mean over words | Co-occurrence baseline |
| Random-init Qwen3.5-2B | same architecture, random weights, seed 0 | Untrained model |
| Pythia-410m step0 | random-init Pythia | Second untrained model |
| Class-blob simulation | each species = its stratum centroid + isotropic Gaussian noise, matched to the real per-class spread | Clusters with no within-class tree |
| Brownian tree simulation | Brownian motion along the reference tree in 2048-d, plus noise | Positive control; power analysis |
| Label permutation | within-stratum or full | Null distribution |

### 4.5 Power
Before any real activation is viewed, a simulation estimates power. Setup:
- the frozen reference tree;
- Brownian-tree signals of varying strength, mixed with class-blob structure and noise;
- the exact analysis pipeline.

The result is the smallest detectable TA_within at 80% power, which is reported in `reports/experiments/00-power/experiment.md`.

---

## 5. Stage 2 probe (preregistered design, exploratory inference)

**Model.** The Goodfire flat-subspace probe: z = W(x − b) with W of shape 10 × 100, applied to PCA-100 of the species vectors.
- PCA is fitted inside each training fold.
- Predicted distance: d̂ = β·arccos(cos(z₁, z₂)).
- Loss: MSE(d̂, d_tree) + α·‖x̂ − x‖², with α = 0.01, where d_tree is the normalized path length.
- Training: Adam, learning rate 1e-3, 2,000 steps, full batch of pairs.

**Cross-validation.** 5-fold, holding out whole families. Metric: Pearson and Spearman correlation on held-out–held-out pairs.

**Controls:**
- the same probe trained on a shuffled tree (leaf labels permuted, shape kept), used to compute selectivity;
- the same probe on random-init Qwen inputs;
- the same probe on GloVe inputs.

**Seeds and dashboard.** Five seeds. Training metrics stream to a live trackio dashboard.

---

## 6. Data exclusion and stopping rules
- **Species-level exclusions,** applied before any activation is extracted:
  - Open Tree match failure;
  - polysemy flag (these go to the sensitivity set);
  - a name span that cannot be located.
- **No data-dependent exclusion** after activations are extracted.
- **Bugs.** If a pipeline bug is found after the main analysis, the analysis is rerun on the fixed pipeline. The bug and both results are reported.

---

## 7. What would change our minds

| Outcome | Interpretation |
|---|---|
| H1 fails, S1 passes | Global correlation is driven by coarse (folk) class clusters: the H_folk shape |
| H1 passes, S3 fails | The within-class signal is largely lexical (H_lex) |
| H1 passes, S4 fails | The within-class signal is explained by co-occurrence (H_cooc) |
| H1, S3 and S4 pass | Phylogenetic structure beyond surface form and co-occurrence, which is the strongest support |
| S5 ≈ 0 | The model encodes Linnaean rank structure, not divergence time |
| Conflict triplets mostly follow the folk partner | The geometry is folk-taxonomic even if it is tree-like |

---

## Deviations
All deviations below were made **before any activation was extracted** (species set frozen 2026-11-20 07:51, `data/FREEZE.json`, species.csv sha256 `f1b24cf4…`).

1. **2026-11-20 — sitelinks threshold.** The registration specified no minimum number of sitelinks for candidates. At ≥15 sitelinks only 4 of 10 echinoderms were available, so the candidate pool uses **≥5 sitelinks**. Ranking is still by sitelinks, so strata already filled at ≥15 are unaffected.
2. **2026-11-20 — reptiles stratum.** OTT flags "Reptilia" as barren and inconsistent. Reptiles are therefore defined as Testudines ∪ Lepidosauria ∪ Crocodylia, which is extant non-avian reptiles, as intended.
3. **2026-11-20 — extinct taxa.** Taxa flagged extinct in OTT are excluded, because TimeTree and the extant tree do not cover them consistently. The thylacine is not flagged and remains.
4. **2026-11-20 — extra polysemy rule.** A name whose English Wikipedia title needed a parenthetical qualifier (e.g. "Beluga (sturgeon)") is flagged as polysemous, because the bare name is ambiguous by Wikipedia's own judgment. This moved "beluga" to the sensitivity set.
5. **2026-11-20 — common-name casing.** Names keep Wikipedia's sentence case for non-initial words. The first word is lowercased unless it is a proper noun: a WordNet capitalized-only lemma, a possessive of an unknown name, or a compass/ocean word before a capitalized word. Six compound words remain capitalized ("Megamouth shark", "Redback spider", …); these are documented in `reports/experiments/learnings.md`.
6. **2026-11-20 — conflict-set placements.**
   - "dog" (Canis familiaris, a subspecies in OTT) is placed at its species, *Canis lupus*.
   - "little brown bat" (*Myotis lucifugus*, a broken taxon in the synthetic tree) is placed at its congener *Myotis myotis*. Every triplet using it splits above genus, so the answers are unchanged.
   - Silverfish's scientific name was corrected to *Lepisma saccharina*.
7. **2026-11-20 — batch size 1 for extraction.** Batched bf16 forward passes on Qwen3.5-2B differ by up to 2.5% relative norm from unbatched ones (not caused by padding), so every prompt is run alone. This is exact and reproducible.
8. **2026-11-20 — S1 reference distance, clarified.** Edge-count path length (S1, as registered) is *not* consistent with rooted triplets (unit test `test_path_length_is_not_triplet_consistent`). S1 is kept as registered. The LCA-depth ultrametric (max depth − depth(LCA)), which is triplet-consistent, is added as an **exploratory** second reference distance for the Mantel analysis.
9. **2026-11-20 — stage order (user request).** Stages were renumbered: causal and behavioral tests (formerly stage 6) are now **stage 3**; ablations → stage 4; sensitivity → stage 5; more models → stage 6. Only the execution order changed. Hypotheses, metrics, tests and multiplicity families are unchanged.
10. **2026-11-20 — exploratory C3b (after the Pythia-410m pilot, before any Qwen activation).** The pilot showed that C3 (overlay length vs *fully* shuffled leaves) passes at every layer, including layer 0 (token embeddings). Any class clustering shortens the overlay, so C3 cannot separate "tree" from "clusters". An exploratory check **C3b** was therefore added: the same statistic against leaves shuffled *within strata*, which tests branching beyond the coarse classes. C3 and the preregistered reading rule are unchanged, and C3b is reported alongside them.
