# Literature review: do LLMs represent a tree of life?

*Libor "Burny" Burian, 2026-11-20. Project: phylogenetic geometry of animal species in LLMs.*

This review collects prior work this project builds on.

- **Scope:** what each source did, what it found (numbers only where the source states them), and what this project takes from it.
- **Status of claims:** every link and number was checked against its primary source in an independent fact-check pass on 2026-11-20 (log at the end). Claims that rest only on an abstract are marked *(abstract only)*.
- **Companion document:** [state-of-the-art-review.md](state-of-the-art-review.md) covers the methods and tools: statistics, data sources, libraries.

---

## 1. Direct precedent: phylogeny in a biological foundation model

### 1.1 Goodfire, "Finding the Tree of Life in Evo 2" (Pearce, Simon, Byun, Balsam; 2025-08-28)

Source: https://www.goodfire.com/research/phylogeny-manifold. The post is the only primary source; no paper, code, or appendix was found.

**Setup**
- Model: Evo 2, Arc Institute's autoregressive DNA language model ([Brixi et al. 2025](https://doi.org/10.1101/2025.02.18.638918)), read at layer 24 of 32.
  - The post does not state the model size. 32 layers matches the 7B config ([evo2-7b config](https://raw.githubusercontent.com/ArcInstitute/evo2/main/evo2/configs/evo2-7b-1m.yml)), so 7B is our inference, not theirs.
  - The layer was chosen by taxon-probe performance.
- Species: more than 2,400 bacterial species from GTDB ([Parks et al. 2022](https://doi.org/10.1093/nar/gkab776)).
- Species vector:
  - Sample random 4 kb genome windows covering about 5% of the genome, drawn independently per species. Independent sampling avoids a sequence-similarity confound.
  - Mean-pool activations over the last 2 kb of each window, so every pooled token has context.
  - Average across windows.

**Ground truth**
- GTDB tree branch-length distance, i.e. patristic distance.

**Geometry analysis** (following Modell et al. 2025, §2.4)
- **Cosine similarity** of the raw embeddings vs phylogenetic distance:
  - mostly monotonic;
  - the figure annotations give Spearman −0.80 and Chatterjee ξ 0.47 (negative because similarity falls as distance grows);
  - the relation flattens at large distances.
- kNN-graph geodesic distance:
  - K=27, the smallest K that gives a connected graph; edges weighted by angular distance;
  - Pearson r = 0.78 with phylogenetic distance (figure annotation);
  - roughly linear except at the largest distances, which are dominated by the outgroup-like order Mycoplasmatales.
- A learned 10-d "flat subspace":
  - model: z = W_enc(x − b), predicted distance d̂ = β·arccos cos(z₁, z₂), plus a small reconstruction loss;
  - cross-validation by holding out ~20 whole clades (~20% of species);
  - distance correlation (figure annotations): r ≈ 0.98 train–train, 0.92 train–test, 0.89 test–test; the text gives the held-out result as "about 0.9";
  - separately, the subspace reconstructs ~70% of embedding variance (29% unexplained on train, 37% on test).

**What the post does not report**
- No permutation tests or p-values, no random-init model, no shuffled-tree null.
- No baseline from k-mer composition alone run through the same geometry pipeline.
- No causal or steering experiments.

**Nucleotide statistics**
- An XGBoost regressor on genome-wide 1–4-mer frequencies predicts the flat-subspace coordinates at r ≈ 0.9. The subspace's first component correlates with TTT, AAT and GTG frequencies (r = 0.74, 0.70, −0.69).
- Goodfire frames this as preliminary evidence that species are encoded as genomic "styles". Reading it as a potential *confound* for phylogeny is our interpretation: genome composition is itself phylogenetically conserved.

**Independent reproduction**
- An ARENA 6.0 capstone (Green & Yukich; [write-up](https://website-co6.pages.dev/writings/evo2), [code](https://github.com/jake-yukich/evo2-mech-interp)) on Evo 2 7B with 2,458 GTDB r220 species.
- Qualitatively similar, quantitatively much weaker: Spearman 0.105, geodesic Pearson 0.379, subspace Pearson 0.504, all train-set numbers.

**Taken for this project**
- The full geometry pipeline: angular distance → kNN geodesic → learned flat subspace with clade holdout.
- The controls the post lacks: permutation, random-init, shuffled tree, confound baselines.

### 1.2 Other biological foundation models *(lower relevance)*
- **Protein language models:** mean-pooled embeddings correlate with LG tree distance ([Briefings in Bioinformatics 2025, PMC11847157](https://pmc.ncbi.nlm.nih.gov/articles/PMC11847157/)). Baselines were one-hot composition and residue shuffling.
- **Audio foundation models on cetacean calls:** Mantel r up to 0.82 against phylogeny, vs 0.04 for an MFCC baseline, with a partial Mantel test controlling for pitch ([arXiv 2607.22458](https://arxiv.org/abs/2607.22458)) *(abstract only)*.
- **BioCLIP 2 (vision):** species embeddings align with ecological traits ([arXiv 2505.23883](https://arxiv.org/abs/2505.23883)) *(abstract only)*.

---

## 2. Neural geometry of concepts in LLMs

### 2.1 Goodfire neural-geometry series
Index: https://www.goodfire.com/research/neural-geometry. The index gives short summaries of six posts; the methods are in the individual posts and papers.

- **"Steering Along Manifolds"** (Wurgaft et al.; [post](https://www.goodfire.com/research/manifold-steering), [arXiv 2605.05115](https://arxiv.org/abs/2605.05115), [code](https://github.com/goodfire-ai/causalab/tree/manifold_steering))
  - Model and readout: Llama-3.1-8B, layer 28, residual stream at the last-token position.
  - Tasks: prompts such as "What day is k days after z?", with weekdays, months, letters and ages.
  - Manifold fitting:
    - PCA to 64 dimensions;
    - per-answer centroids;
    - a periodic cubic spline through the centroids for cyclic concepts, a natural spline for sequential ones.
  - Intervention: steer along the spline, keeping the base prompt's off-subspace component, compared with a straight chord between centroids.
  - Results:
    - isometry (Pearson between distance matrices) of geodesic vs behavior: 0.99 for weekdays, 0.89 for months, vs 0.89 and 0.53 for linear distance;
    - manifold steering lowers a cumulative Bhattacharyya-energy proxy for distance from the model's natural output distribution by about 2.8× on average vs linear steering (p < 0.001).
  - Controls: no random or shuffled control.
- **"A Geometric Calculator"** (Feucht et al.; [post](https://www.goodfire.com/research/a-geometric-calculator), [arXiv 2605.01148](https://arxiv.org/abs/2605.01148)): Llama-3.1-8B adds numbers using circular representations at several moduli *(post only)*.
- **"Can SAEs Capture Neural Geometry?"** (Bhalla et al.; [post](https://www.goodfire.com/research/can-saes-capture-neural-geometry); paper "Do Sparse Autoencoders Capture Concept Manifolds?", [arXiv 2604.28119](https://arxiv.org/abs/2604.28119)): SAE features fragment manifolds, a "dilution" regime.

### 2.2 Engels et al. 2024, "Not All Language Model Features Are One-Dimensionally Linear"
Sources: [arXiv 2405.14860](https://arxiv.org/abs/2405.14860), [code](https://github.com/JoshEngels/MultiDimensionalFeatures).

- **Discovery:** cluster SAE decoder directions in GPT-2-small and Mistral-7B. Days of the week and months appear as circles in PCA of the cluster's reconstructions. PC1 often encodes radius; PCs 2–3 show the circle.
- **Intervention:** a circular probe on the top-5 PCs at the day token, then patching in a target angle. Measured by logit difference, against baselines that patch PCs, the full layer, or the task mean.
- **Taken for this project:** the "top-PCA plot first" methodology (stage 1a) and the days/months positive control.

### 2.3 Modell, Rubin-Delanchy, Whiteley 2025, "The Origins of Representation Manifolds in Large Language Models"
Sources: [arXiv 2505.18235](https://arxiv.org/abs/2505.18235), [code](https://github.com/alexandermodell/Representation-Manifolds).

- **Theory:** features are metric spaces, explicitly including hierarchical (tree) metrics. If cosine similarity is locally a decreasing function of feature distance, then geodesic length on the representation manifold is proportional to feature distance.
- **Tests:**
  - Chatterjee's ξ for cosine vs feature distance;
  - Pearson between kNN-graph geodesic distance (smallest connected k) and feature distance;
  - examples: years, dates, colours.
- **Taken for this project:** this is the theoretical justification for testing geodesic vs angular distance against a tree metric.

### 2.4 Park, Choe, Jiang, Veitch 2024/25, "The Geometry of Categorical and Hierarchical Concepts in Large Language Models"
Sources: [arXiv 2406.01506](https://arxiv.org/abs/2406.01506), [code](https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations).

- **Representation space:** whitened unembedding vectors of Gemma-2B (LLaMA-3-8B in the appendix), using a "causal inner product".
- **Concepts:** WordNet synsets, with an LDA-like concept-vector estimator.
- **Findings:**
  - hierarchically related concepts are orthogonal, i.e. parent ⟂ (child − parent);
  - categories form simplices;
  - the animal sub-categories used in figures were generated with GPT-4 and then checked, not taken from WordNet.
- **Critique:** a 2025 ICLR blog post, ["Intricacies of Feature Geometry in LLMs"](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-feature-geometry-65/blog/feature-geometry/), finds the same orthogonality and polytopes for random unembeddings and nonsense categories under the same whitening.
- **Taken for this project:** whitening is never the primary analysis, and random-init and nonsense/shuffled controls are mandatory.

### 2.5 Gurnee & Tegmark 2023, "Language Models Represent Space and Time"
Sources: [arXiv 2310.02207](https://arxiv.org/abs/2310.02207), [code](https://github.com/wesg52/world-models).

- **Method:**
  - residual stream at the last token of an entity name, at every layer;
  - ridge probes for coordinates or dates;
  - block holdouts, e.g. whole countries;
  - Llama-2 and Pythia models.
- **Result:** Llama-2-70B at about 60% depth reaches world-places R² 0.911. Representations are roughly linear; MLP probes give little gain.
- **Taken for this project:** the last-name-token readout, the all-layer sweep, and block (clade) holdout.

### 2.6 Co-occurrence explanations of hierarchical and cyclic geometry
- **Karkada et al. 2026** ([arXiv 2602.15029](https://arxiv.org/abs/2602.15029), [code](https://github.com/dkarkada/symmetry-stats-repgeom)): translation symmetry in co-occurrence statistics produces circles and lines in word embeddings and in Gemma-2-2B residuals.
- **Nava & Wyart 2026** ([arXiv 2605.23821](https://arxiv.org/abs/2605.23821)):
  - WordNet hypernym distance predicts co-occurrence;
  - the top Gram-matrix eigenvectors then split the tree from coarse to fine in word2vec and in whitened Gemma-2B unembeddings;
  - co-occurrence alone reproduces the Park et al. near-orthogonality.
- **Taken for this project:** H_cooc, the hypothesis that co-occurrence statistics explain the geometry, is a primary alternative explanation. We control for it with static-embedding baselines and, for Pythia, Pile counts.

### 2.7 SAE feature geometry
- **Li, Michaud, Baek, Engels et al. 2024, "The Geometry of Concepts"** ([arXiv 2410.19750](https://arxiv.org/abs/2410.19750), [code](https://github.com/ejmichaud/feature-geometry)): Gemma-2-2B with Gemma Scope SAEs ([Lieberum et al. 2024](https://arxiv.org/abs/2408.05147)). Finds "crystals" (after LDA removes distractors such as word length), functional "lobes", and power-law PCA spectra.
- **Kissane et al. 2024** ([LessWrong](https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models)): SAEs trained on base models usually transfer to chat models. This was tested on Mistral-7B and Qwen1.5-0.5B, and transfer **failed** for Gemma v1 2B. It matters here because the cached Qwen3.5 SAE was trained on the Base model; whether it transfers to the hybrid Qwen3.5 architecture is an extrapolation that must be checked.

### 2.8 Tree-structured representations
- **Baek, Li, Tegmark 2024** ([arXiv 2410.08255](https://arxiv.org/abs/2410.08255)): an in-context family tree is recovered with a "cone probe" on 10-d PCA of residuals, with subspace patching as the causal test.
- **Hewitt & Manning 2019, structural probe** ([NAACL](https://aclanthology.org/N19-1419/)): a linear map B with ‖B(hᵢ−hⱼ)‖² ≈ tree distance, applied to syntax trees. Goodfire's flat subspace is the angular analogue.
- **Hewitt & Liang 2019, control tasks** ([EMNLP](https://aclanthology.org/D19-1275/)): selectivity is probe accuracy minus control-task accuracy. Basis of our shuffled-tree probe control.
- **Chen et al. 2021, Poincaré probe** ([arXiv 2104.03869](https://arxiv.org/abs/2104.03869), [code](https://github.com/FranxYao/PoincareProbe)): hyperbolic probes recover syntax trees in BERT better than Euclidean probes.

---

## 3. Hyperbolic / tree-like geometry
- **Nickel & Kiela 2017, Poincaré embeddings** ([arXiv 1705.08039](https://arxiv.org/abs/1705.08039), [code](https://github.com/facebookresearch/poincare-embeddings)): embeds the WordNet noun hierarchy in hyperbolic space with low distortion; the mammal subtree is shown in a 2-D visualization.
- **Sala, De Sa, Gu & Ré 2018 (ICML), representation tradeoffs for hyperbolic embeddings** ([arXiv 1804.03329](https://arxiv.org/abs/1804.03329), [code](https://github.com/HazyResearch/hyperbolics)): combinatorial construction and h-MDS, with distortion and MAP metrics.
- **Hyperbolicity Atlas** ([arXiv 2609.07053](https://arxiv.org/abs/2609.07053)):
  - four-point δ_rel = δ/diam over sampled quadruples of hidden states;
  - middle layers sit near δ_rel ≈ 1 (median ≈ 0.995 for Qwen2.5-7B-Base), while final layers are more tree-like;
  - controls are one random-init model and a prompt-length-quartile analysis.
- **HypLoRA** ([arXiv 2410.04010](https://arxiv.org/abs/2410.04010)): frequent tokens sit nearer the origin. Token frequency therefore confounds norm-based hierarchy *(abstract only)*.
- **Chourasia et al.** ([arXiv 2512.20926](https://arxiv.org/abs/2512.20926)): a toolkit combining δ-hyperbolicity, ultrametricity and neighbor-joining metrics for protein LMs *(abstract only)*.

**Caveat for δ-hyperbolicity.** In high dimensions, pairwise distances concentrate, and nearly equidistant random points look star-tree-like. So δ_rel must be compared against a spectrum-matched Gaussian null and against non-hierarchical concept sets. Label permutation does not change δ, so it is not a valid null for it.

---

## 4. Language models, species, and taxonomy

**Gap.** No peer-reviewed study was found that tests whether text-LLM hidden states for species names reflect the phylogenetic tree of life. Searches covered arXiv, Semantic Scholar via Consensus, GitHub and Hugging Face, as of 2026-11-20.

Closest work:
1. **phylotext-mammals** (student project, preregistered, results locked 2026-11-11; [repo](https://github.com/jadencutinha/phylotext-mammals))
   - Method: sentence-embedding models (bge-large, mpnet) applied to Wikipedia descriptions with the species name masked, compared with the Upham 2019 mammal tree using Mantel tests. Partial Mantel and MRM control for EltonTraits ecology and for pageviews or article length.
   - Results:
     - 4,611 mammals: Mantel r = −0.028, not significant;
     - within-order r is positive for some orders (Artiodactyla 0.28, Primates 0.14);
     - leaving common-name head nouns unmasked raises r to about 0.19.
   - Main lessons: popularity and head-noun leakage are the key confounds.
2. **llm_internal_representations** ([repo](https://github.com/alexgarciaduran/llm_internal_representations)): 87 organisms in mBERT, GPT-2 and Qwen2.5-0.5B, all layers, Mantel test against taxonomic rank depth. ρ ≈ 0.36–0.39. Rank-based ground truth only, with no dated tree and no confound controls.
3. **Taxonomy question answering:**
   - LLMs classify IUCN species from kingdom to family in a multiple-choice format at 94.9% accuracy overall: 91.8% with phylogenetically close distractors vs 98.0% with random ones (Uryu 2025, [arXiv 2510.02830](https://arxiv.org/abs/2510.02830));
   - the BAGEL animal-knowledge benchmark ([arXiv 2604.16241](https://arxiv.org/abs/2604.16241)) *(abstract only)*.
   - These test labels, not geometry.
4. **Grand et al. 2022** ([Nature Human Behaviour](https://doi.org/10.1038/s41562-022-01316-8)): projecting GloVe vectors onto semantic axes recovers animal size and dangerousness. Folk-ecological dimensions like these are confounds for phylogeny.

---

## 5. Synthesis: hypotheses and open questions

**Hypotheses the literature motivates**

| Hypothesis | Motivation | Discriminating test in this project |
|---|---|---|
| H_folk: only folk categories (bird / mammal / fish / bug) | Class clusters dominate any global correlation; phylotext found ~0 global signal | Within-class triplet accuracy; phylogeny-vs-folk conflict triplets |
| H_lex: surface form (shared head nouns) | phylotext: head nouns raised r from ~0 to 0.19 | Head-noun-disjoint and lexical-discordant triplets; char-n-gram baseline |
| H_cooc: co-occurrence statistics | Nava & Wyart 2026; Karkada et al. 2026 | Static-embedding baseline; co-occurrence-discordant triplets; Pythia + Pile counts |
| H_freq: popularity/frequency | HypLoRA; phylotext | Frequency covariates; PC-vs-frequency correlations |
| H_rank vs H_time: Linnaean ranks vs dated phylogeny | Text discusses ranks more often than divergence times | Rank–time discordant pairs |
| H_geom: tree / manifold geometry à la Goodfire | Goodfire Evo 2; Modell et al. | Geodesic vs angular distance beyond the class-blob null; δ_rel vs matched nulls; flat probe with family holdout |

**Open questions this project addresses**
1. Is there phylogenetic geometry in a text LLM beyond class-level folk categories?
2. Does it survive lexical, frequency and co-occurrence controls?
3. Is it tree-shaped (geodesic or hyperbolic) or just clustered?
4. Is it causally used by the model?

---

## Fact-check log
- 2026-11-20: first draft, compiled from three literature-search passes, each reading primary sources.
- 2026-11-20: independent fact-check of every link and number. All DOIs, arXiv ids and ACL ids resolve to the stated works. Corrections applied:
  - Goodfire's −0.80 is for cosine *similarity*, not angular distance.
  - The ~70% is reconstructed embedding variance, not the distance correlation.
  - Goodfire frames the composition result as "genomic style"; the confound reading is ours.
  - Engels et al. title.
  - Manifold steering: readout position and the meaning of 2.8×.
  - Chen et al. claim narrowed; Nickel & Kiela scope corrected.
  - Hyperbolicity Atlas controls; Kissane et al. scope.
  - arXiv 2510.02830 details.
- Full report: [reviews/2026-11-20-fact-check-literature.md](reviews/2026-11-20-fact-check-literature.md).
