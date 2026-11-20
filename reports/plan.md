# Plan: Phylogenetic geometry of the tree of life in LLMs

## Context

**Goal.** Test whether open-weight LLMs represent animal species with tree-of-life geometry, and characterize that geometry. Two Goodfire works serve as templates.

- **"Finding the Tree of Life in Evo 2"** ([post](https://www.goodfire.com/research/phylogeny-manifold), Pearce et al. 2025-08-28)
  - Setup: Evo 2 layer 24/32, mean-pooled per-species activations, 2,400 GTDB bacteria.
  - Angular distance vs GTDB patristic distance: Spearman −0.80.
  - kNN-graph geodesic (K=27, the smallest K that gives a connected graph): Pearson 0.78.
  - Learned 10-d "flat subspace": r≈0.98 on train, 0.89 on held-out clades.
  - Missing from the post: code, permutation tests, random-model or shuffled-tree controls, causal tests.
  - An independent ARENA reproduction got much weaker numbers (geodesic r 0.38).
- **Goodfire neural-geometry series** ([index](https://www.goodfire.com/research/neural-geometry))
  - Covers days-of-week circles and manifold steering in Llama-3.1-8B ([arXiv 2605.05115](https://arxiv.org/abs/2605.05115)).
  - The underlying theory: [Modell et al. 2025](https://arxiv.org/abs/2505.18235) show geodesic ∝ feature distance, explicitly including tree metrics.

**Gap.** No peer-reviewed study tests text-LLM hidden states for species names against the phylogeny. The closest work:
- A student project ([phylotext-mammals](https://github.com/jadencutinha/phylotext-mammals)) found sentence embeddings uncorrelated with the mammal tree, with confounding from popularity and head nouns.
- A small 87-organism repo ([llm_internal_representations](https://github.com/alexgarciaduran/llm_internal_representations)).

**Strongest alternative explanations.** These come from the literature and the design critique:
1. **Folk categories:** class-level clusters (birds vs mammals) inflate any global correlation.
2. **Lexical leakage:** shared head nouns, as in "X shark".
3. **Co-occurrence statistics:** [Nava & Wyart 2026](https://arxiv.org/abs/2605.23821) show co-occurrence alone produces hierarchical splits.
4. **Frequency/popularity.**
5. **Whitening and high-dimensional artifacts:** the [Park et al.](https://arxiv.org/abs/2406.01506) critique, and δ-hyperbolicity on noise.

The design is built to *discriminate* these explanations from genuine phylogenetic geometry.

**Project state.** The project folder is empty, so this is a whole-project plan. It follows the user's global CLAUDE.md order: literature review → preregistration pushed before any result → run matrix → src+tests pushed → staged runs, pushing to GitHub and HF after each → reviews → paper and site.

## Research questions & preregistered hypotheses

The full text goes in `reports/preregistration.md`.

**Primary metric: node-balanced rooted-triplet accuracy (TA).**
- A triplet (a,b|c) qualifies when the reference tree resolves it, i.e. MRCA(a,b) lies strictly below MRCA(a,b,c). Polytomies are excluded.
- The triplet is correct if d(a,b) < min(d(a,c), d(b,c)). Chance is 1/3.
- At most K triplets are sampled per decision node, so large clades do not dominate.
- Properties: needs topology only, is invariant to monotone transforms, and gives a **depth profile** of TA by the rank of the decision node.

**Single primary test.**
- Statistic: `TA_within`, i.e. triplets whose decision node is below class rank and whose three head nouns all differ.
- Setup: Qwen3.5-2B (the standard post-trained release, not Base), mean-centered cosine distance.
- Significance: max-T over layers with a within-class label-permutation null.
- Point estimate: family-blocked 2-fold cross-fitting for layer choice.
- Smallest effect of interest: TA ≥ 0.40. Falsified if the CI includes 1/3, or if TA does not exceed the random-init Qwen3.5-2B, Pythia step0 and the lexical baselines.

**Confirmatory secondaries** (Holm-corrected):
1. Global Spearman/Kendall-τb Mantel ρ, for comparability with Goodfire.
2. Order-level depth TA.
3. TA on lexical-discordant triplets (character 3-gram distance picks the wrong answer).
4. TA on co-occurrence-discordant triplets (static embeddings or Pile PMI pick the wrong answer).
5. H_rank vs H_time on rank–time discordant pairs. Example: insect inter-order splits (~350–400 Mya) vs mammal inter-order splits (~90 Mya).

All other analyses are exploratory and BH-corrected.

**Discriminating predictions:**

| Hypothesis | Prediction that distinguishes it |
|---|---|
| H1a (stage 1a, visual) | In the top-3 PCs, species form a branching tree: clade arms with nested sub-branches, and a tree-overlay length shorter than the shuffled-label null. The alternatives are blob, separate clusters, gradient, curve, or frequency-driven structure (details in Stage 1a). |
| H_folk | High global ρ, but TA_within ≈ 1/3; conflict triplets follow folk groupings (whale→shark, bat→bird, horseshoe crab→crab) |
| H_lex | TA collapses on head-noun-disjoint and lexical-discordant triplets; ≈ char-n-gram, input-embedding and step0 baselines |
| H_cooc | Model TA ≈ static-embedding TA; ≈ chance on co-occurrence-discordant triplets |
| H_rank / H_time | TA_within ≫ 1/3 and survives discordant subsets; on rank–time discordant pairs, geometry follows Linnaean rank or dated time |
| H_geom (Goodfire) | Geodesic r − angular r exceeds the class-blob null; δ_rel < matched-spectrum Gaussian and < non-hierarchical concept sets; flat probe generalizes to held-out families with selectivity over a shuffled tree; Poincaré beats Euclidean probe at d ≤ 10 |
| H_causal | Ablating the probe subspace hurts taxonomic prompts more than a variance-matched random in-span subspace, with small off-target loss change; subspace interchange switches taxonomic answers more than a random subspace |

## Data

**Species: about 400 animals, frozen and hashed (sha256) before any activation is viewed.**
- Source: Wikidata SPARQL. Criteria: species rank, in Animalia, an English Wikipedia article whose title is the common name, an NCBI taxid (P685), and an Open Tree OTT match.
- Stratified by class/phylum: mammals, birds, reptiles, amphibians, ray-finned fish, cartilaginous fish, insects, arachnids, crustaceans, molluscs, cnidarians, echinoderms, other.
- Within each stratum, the top species by English Wikipedia pageviews are taken, capped at **3 per family**, with **at least 50% single-word names**.
- Polysemy filter: drop names that have a non-animal WordNet sense or a disambiguation page (jaguar, python, seal, crane, …). Kept only for a sensitivity analysis.
- Dialect-ambiguous names (e.g. robin) are dropped.

**Conflict triplets: about 40, hand-curated and preregistered.** Every answer is computed from the tree.
- Phylogeny vs folk: whale/hippo/shark, crocodile/sparrow/lizard, tuna/human/shark, starfish/human/octopus, horseshoe crab/spider/crab, bat/cow/sparrow, hyrax/elephant/guinea pig, thylacine/kangaroo/wolf, lamprey/eel/human, lobster/beetle/spider.
- Lexical-conflict names: starfish, jellyfish, killer whale, sea lion, flying fox, silverfish.

**Reference trees.**
- **Primary (redistributable, CC0):** Open Tree induced-subtree topology ([license](https://tree.opentreeoflife.org/about/licenses)), annotated with NCBI ranks via ete4. Broken or pruned taxa are logged.
- **Dated tree:** TimeTree 5 MRCA ages ([API](https://timetree.temple.edu/api), by NCBI ids), one call per internal node, giving an ultrametric patristic distance.
  - TimeTree data may not be redistributed. It is cached in gitignored `DATA_DIR`, and only statistics and the refetch code are published.
- **Oracle baselines:** a class oracle (13 labels) and an NCBI-rank oracle.

**Confounder covariates.**
- Character 3-gram distance, head noun, token count.
- Frequency: infini-gram counts ([API](https://infini-gram.readthedocs.io/en/latest/api.html); `v4_piletrain_llama` for Pythia) and Wikipedia pageviews.
- Static embeddings (fastText/GloVe) and the model's own input embedding.
- Habitat: coarse, from Wikidata, with TetrapodTraits for tetrapods (stage 2).

## Representations

- **Prompts:** 8 templates, all with the same animal-priming context, the article "the", and no class words.
  - Qwen3.5-2B is a chat model, so the primary format wraps each template in the official chat template as the user turn, with thinking disabled (`enable_thinking=False`).
  - Raw text without the chat template is a stage-3 ablation.
  - Pythia gets raw text with BOS prepended, since it omits BOS by default.
- **Name span:** located by tokenizer offset mapping and tested under left and right padding.
- **Readout:** residual stream at every layer.
  - Primary: last name token.
  - Secondary: mean over name tokens.
- **Normalization (pre-specified):** subtract the species-mean per template, average over templates, then cosine distance. Sensitivity checks: per-dimension z-scoring and raw cosine. Whitening is never primary.
- **Noise ceiling:** split-half template reliability.
- **Numerics:** exact bf16, with the whole model on the GPU. No CPU offload and no quantization (quantization perturbs geometry and bitsandbytes is fragile on sm_120).
  - [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B): Apache-2.0, not gated, 2.27B parameters, about 4.6 GB in bf16. It is a fine-tune of Qwen3.5-2B-Base.
  - Architecture: 24 layers, d=2048, a hybrid of 3 Gated-DeltaNet linear-attention blocks per full-attention block.
  - It needs a one-time download to `D:\AI\hf_cache`.
  - Load the text decoder only. The checkpoint ships as multimodal `Qwen3_5ForConditionalGeneration`, so verify the hidden states against the full model.
  - If no fast linear-attention kernel is available on Windows, use the torch fallback. That is fine for about 3.2k short prompts.
- **Storage:** fp32 readouts per template and layer stored as `.npy` in `DATA_DIR` (D:).

## Models and staged run matrix

The full matrix goes in `reports/planned-runs.md` and the README.

**Stage 1a: simple PCA look (done first, per the user, as in Goodfire neural geometry).** This is the first experiment run. It is declared in the prereg as exploratory and visual; confirmatory tests are fixed before these plots are seen.
- **Data:** Qwen3.5-2B residual-stream activations at the last name token, for every species × 8 prompt templates, at every layer (24 layers).
- **Plot:** for each layer, mean-center the vectors, fit PCA, and plot the **top 3 PCA directions**:
  - Interactive 3D Plotly scatter: species = big dots colored by phylum/class, hover shows names; each individual prompt = a small dot, showing prompt variability.
  - 2D panels: PC1–2, PC2–3, PC1–3, with explained variance on the axes.
  - Layer slider/grid.
- **Tree overlay:** place each reference-tree internal node at the mean PCA position of its descendant leaves and draw the tree edges in PC space. This shows directly whether branches fan out like a tree.
- **Zoom-ins:** PCA within single classes (mammals, birds, insects) to look for sub-branches such as orders.
- **Calibration plots, same code:**
  - Days of week / months, expecting the known circle ([Engels et al.](https://arxiv.org/abs/2405.14860)).
  - Random-init Qwen3.5-2B (negative control).
  - Synthetic Brownian-motion-on-tree point cloud (what a true tree looks like in the top 3 PCs).
- **Hypothesis H1a ("do they look like a tree?")**, preregistered and labeled exploratory: in some layer's top-3 PCs, species form a **branching tree shape**.
  - What a tree looks like:
    - Major clades (phyla, then classes) occupy distinct arms or rays from a common center, rather than separate islands.
    - Sub-clades (orders) split off *inside* their parent arm, giving nested branching.
    - The tree overlay drawn in PC space is compact with few long crossing edges.
    - Calibrated against the synthetic Brownian-tree cloud plotted with the same code.
  - Alternative shapes it must be distinguished from:
    - (a) an isotropic blob, i.e. no structure;
    - (b) separated class clusters with no internal branching, the folk-category shape (H_folk);
    - (c) a continuous gradient along an ecological or size axis, e.g. aquatic↔terrestrial;
    - (d) a curve or circle, as for days of the week;
    - (e) frequency-driven structure, with PC1 tracking log-frequency or token count.
  - Pre-specified quantitative checks, run with the plots and using the permutation and random-init Qwen nulls:
    1. Class separation in PC1–3: kNN class purity and silhouette, vs a label-permutation null and vs random-init Qwen.
    2. Nested branching: order-level kNN purity in within-class PCA (mammals, birds, insects), vs a within-class permutation null.
    3. Tree-overlay length: the summed edge length of the reference tree with internal nodes at descendant centroids in PC1–3, vs the same quantity under shuffled leaf labels. This is a squared-change-parsimony-style multivariate phylogenetic-signal statistic, in the spirit of [Blomberg's K](https://doi.org/10.1111/j.0014-3820.2003.tb00285.x). Shorter than the null means tree-consistent.
    4. What each PC encodes: correlation of PC1–3 with class indicators, log-frequency, token count and coarse habitat.
  - Reading the outcome:
    - **"Looks like a tree"** requires checks 1, 2 and 3 to all beat their nulls in the same layer, with arms rather than islands in the plot.
    - **"Clusters, not a tree"**: check 1 passes but checks 2 and 3 fail.
    - **"Confound-driven shape"**: the top PCs track frequency, token count or habitat more than clade.
    - **"No structure"**: check 1 fails in every layer.
  - Caveat: the top 3 PCs hold only part of the variance. A negative result here does not falsify a tree in higher dimensions; stage 1b (TA, geodesic) tests that.
  - Anti-bias step: first describe the shape from an **uncolored** plot, and only then color by clade and add the tree overlay. Record both descriptions.
- **Record:** write qualitative observations (shape, branching, which layers, which clades separate on which PC) plus the H1a check results to `reports/experiments/01-pca-shape/experiment.md` before any quantitative stage-1b analysis.
- **Output:** PNG and HTML in `results/` and the site.
- **Code:** `viz/pca.py`, with the `pca` phase called by `run.py`.

**Stage 1b: quantitative main method, no training (≈1 day).**
- Models:
  - **Qwen3.5-2B** (primary, per the user's choice: the standard post-trained release, not Base). It fits fully on the 8 GB GPU in bf16.
  - Qwen3.5-2B with **random-init weights** (`from_config`, fixed seed). This is the architecture-matched untrained control.
  - Pythia-410m (dev/pilot model, declared in the prereg) and Pythia-410m `step0` (second random-init control).
- Baselines and controls:
  - Class oracle, rank oracle, character 3-gram distance, input embeddings.
  - Class-blob simulation: class centroids plus isotropic noise, run through the same pipeline.
  - Within-class label permutation.
  - Synthetic Brownian-motion-on-tree point cloud (positive control and power simulation).
- Outputs:
  - TA depth profile × layer vs baselines.
  - TA_within with a clade-block bootstrap CI (resample families; drop same-origin pairs).
  - Global Mantel with max-T.
  - Noise ceiling and the lexical-discordant subset.

**Stage 2: trained probes and controls.** Following Goodfire's flat-subspace model:
- Model: z = W(x−b), d̂ = β·arccos cos(z₁,z₂), with a reconstruction term.
- Training: PCA-100 fit inside each fold; whole families held out.
- Controls: a shuffled-tree probe (leaf labels permuted, tree shape kept) for selectivity, plus probes on random-init inputs and on static-embedding inputs.
- Training metrics are logged to a **trackio** live dashboard.
- Geometry: geodesic vs angular (minus the class-blob null gap).
- Tree-likeness: δ_rel (Fournier algorithm, checked against brute force) vs a matched-spectrum Gaussian and vs non-hierarchical concept sets (colors, numbers). Neighbor-joining tree from the representations vs the reference by triplet distance and RF, against a random-tree null.

*Stage order changed by the user on 2026-11-20: causal/behavioral moved from 6 to 3; ablations, sensitivity and more models shift to 4, 5, 6.*

**Stage 3: causal and behavioral.**
- Mean-ablation of the probe subspace vs a variance-matched random in-span subspace, on held-out species.
  - Target: taxonomic-prompt log-probs, e.g. "The {name} is a kind of" → mammal/bird/….
  - Off-target: loss on neutral text.
- Interchange intervention (swap subspace coordinates between species) → answer-switch rate.
- Exploratory steering along the geodesic vs the linear chord ([Wurgaft et al.](https://arxiv.org/abs/2605.05115)).
- Behavioral triplet judgments by log-prob, in both answer orders.
- Positive control: days/months circle replicated through the same geodesic pipeline.
- Exploratory: SAE features for the phylogeny subspace.
  - Use them only if an SAE suite exists for post-trained Qwen3.5-2B (check Qwen's "QwenScope" release).
  - The cached SAE was trained on the Base model, so it gets a transfer check before any use ([Kissane et al. 2024](https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models)).

**Stage 4: ablations.**
- Layer, readout, template subsets, distance metric, kNN K, and subspace dimension.
- Common names vs scientific binomials, using triplets split above genus only.
- Including vs excluding polysemous names.
- Dated tree: H_rank vs H_time.

**Stage 5: sensitivity.** Probe seeds (5), template-subset resampling, species subsampling, and kNN/dimension sweeps.

**Stage 6: more models.** All fit on the GPU in bf16.
- Qwen3.5-0.8B (post-trained): scale within the family.
- Qwen3-1.7B (post-trained): standard full-attention Qwen architecture, checking whether the hybrid linear attention matters.
- Optional extra families if time allows: gemma-4-E2B, SmolLM3-3B.
- Pythia 70m→2.8B for scaling.
- Pythia-1.4B at about 12 log-spaced checkpoints for training dynamics, paired with infini-gram Pile frequency and co-occurrence counts (H_cooc).

Deferred or cut: structural probe (redundant with the flat probe), models that do not fit on the GPU (Qwen3.5-4B/9B, Llama-3.1-8B), pageview weighting.

## Architecture

Package `src/phylogeo/`, managed with uv. One entry point:

```
uv run python -m phylogeo.run --config configs/stage1.yaml --phases species,tree,extract,analyze,report
```

Phases are idempotent, cached by config hash, and recorded in `manifest.json`.

- `config.py`: dataclasses plus YAML; paths read from `.env`; seeds.
- `data/species.py`: Wikidata query, pageviews, polysemy filter, family cap, freeze+hash.
- `data/tree.py`: Open Tree match and induced subtree, NCBI lineage and ranks via ete4, MRCA table, TimeTree fetch (cached), patristic and rank distances.
- `data/baselines.py`: character n-gram, head noun, input and static embeddings, class oracle, class-blob simulation, infini-gram counts.
- `prompts.py`: templates, rendering, name-token span.
- `extract.py`: model loading (bf16 on GPU, plus random-init variant), `extract_name_reprs` → `.npy`.
- `geometry.py`: centering, cosine distance, kNN geodesic, smallest connected K.
- `stats/mantel.py` (Mantel, partial Mantel, max-T, restricted permutations); `stats/triplets.py` (decision nodes, sampling, TA, depth profile, discordant subsets); `stats/resample.py` (clade bootstrap).
- `treeness.py`: δ-hyperbolicity, NJ via scikit-bio, triplet distance, RF via DendroPy.
- `probes/flat.py`, `probes/cv.py`; `causal/ablate.py`, `causal/interchange.py`; `behavior.py`, which logs every prompt and output to `logs/`.
- `viz/pca.py`: top-3-PCA scatter (3D HTML plus 2D PNG panels), tree-edge overlay, layer slider, and calibration sets (days, months, random-init, synthetic tree).
- `report/figures.py`, `report/tables.py`, `report/site.py`: every number, table and figure is regenerated from `results/` by a single `--phases report`.

**Reused libraries rather than reimplemented:**
- scikit-bio `mantel` and `nj`; DendroPy; ete4; scipy/sklearn graph shortest paths.
- skdim (intrinsic dimension); geoopt (Poincaré probe); transformers/accelerate; nnsight (causal hooks); trackio; Plotly.
- Typst via the `typst` PyPI package (paper PDF).

Partial Mantel is the one hand-written method, since no maintained Python package exists. It is tested against the closed-form partial correlation and synthetic data.

## Tests (TDD, pytest; written before the modules)

- **Mantel:** equals `skbio.mantel` (Pearson and Spearman); p-values uniform under the null (KS test over 200 simulations); max-T controls FWER.
- **Partial Mantel:** ≈0 when the association is fully mediated; >0 when there is a direct effect.
- **TA:** 1.0 on an exact tree metric; ≈1/3 on random vectors; polytomies excluded; node weights correct.
- **Tree:** patristic distances match DendroPy; ultrametric check; pruning preserves distances.
- **Tree-likeness:** δ=0 on tree metrics; known δ on the cycle graph C_n; fast δ equals brute force at n=20; NJ recovers an additive tree with RF=0.
- **Geometry:** geodesic on a noisy circle gives ρ>0.99; smallest-connected-K correct on two blobs.
- **PCA viz:** a synthetic noisy circle embedded in 2048-d recovers cyclic order in PC1–2. The tree-overlay node positions equal the mean of each node's descendant leaves.
- **Tree-overlay length:** shorter than the shuffled-label null on the synthetic Brownian tree; ≈ null on random vectors. kNN purity ≈ chance on random labels.
- **Bootstrap:** drops self-pairs and resamples whole clades.
- **Extraction:** the text-decoder hidden states equal the full `Qwen3_5ForConditionalGeneration` forward pass; batched output equals unbatched under padding (important for linear-attention state); name span correct inside the chat template, under both paddings, and with multi-token names; BOS present for Pythia.
- **Species:** schema, unique taxids, every species in the tree, family cap holds.
- **Probes:** recovers a planted 10-d tree; shuffled-tree control ≈0 on held-out species.

## Environment & infrastructure

- **Qwen3.5-2B download:** about 4.6 GB, fetched once with `hf download Qwen/Qwen3.5-2B` into `D:\AI\hf_cache`.
- **Python:** `uv init`, Python pinned to 3.12 (system 3.15 lacks torch wheels), local `.venv`, PyTorch cu128+ build. Check that `sm_120` is in `torch.cuda.get_arch_list()`.
- **Disk:** C: has only 15 GB free. `.env` sets `HF_HOME=D:\AI\hf_cache` (models already cached there), `UV_CACHE_DIR=D:\AI\uv_cache`, `UV_LINK_MODE=copy`, and `DATA_DIR=D:\AI\phylo-geometry`.
- **Logging:** timestamped logs in `logs/` (Python logging to both file and terminal).
- **GitHub:**
  - Public repo `BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability`, public because GitHub Pages on the free plan needs it.
  - Feature branches, PRs and self-merge; commits split by function; no credentials (`.env` gitignored, `.env.example` committed).
  - No AI attribution in commits.
- **Hugging Face:** one collection, "Phylogenetic geometry of the tree of life in LLMs".
  - Datasets: the species dataset (names, taxids, OTT ids, Open Tree topology, ranks, frozen hash) and the activations (Qwen and Pythia, both Apache-2.0).
  - Models: probe weights.
- **Site:** static Plotly.js explorer in `site/`, deployed with a GitHub Actions Pages workflow.
  - Features: a 3D PCA/Isomap species cloud with a layer slider, colored by clade, with hover names; reference tree vs NJ tree; TA depth profile; conflict triplets.

## Deliverables (per the user's CLAUDE.md)

- **Literature reviews:** `reports/literature-review.md` and `reports/state-of-the-art-review.md`, built on the three research-agent reports, cited and then fact-checked.
- **Preregistration:** `reports/preregistration.md`, pushed before any results.
- **Run matrix and plan:** `reports/planned-runs.md`, `reports/plan.md`.
- **Ongoing records:** `reports/progress.md` (updated periodically, with status reported in chat); `reports/experiments/experiments.md` and `learnings.md`, plus one subfolder per experiment; `reports/reviews/` (NeurIPS-style red-team reviews after implementation, after stage 2, and at the end).
- **Paper and summary:** `reports/paper.pdf` and `reports/summary.md`.
  - The paper credits Libor "Burny" Burian as author and is dated.
  - It includes an architecture diagram, eval table, example visualizations, all citations, and an AI statement at the bottom (the README gets no AI statement).
- **README:** top summary, figures, methods, how to run, results table in plain language, data flow, features and tech, TODO. Overflow goes to `docs/` and `AGENTS.md`.
- **Data:** raw results in `results/`; large arrays on D: and on HF.

## Execution order

1. Initialize the project: `uv init`, .env, .gitignore, gh repo create.
2. Literature and SOTA reviews, then a fact-check pass.
3. Preregistration and planned runs, plus README; push to GitHub.
4. Write tests first (TDD), then the core src; push.
5. Build and freeze the species list and tree (hash). Run initial tests and a Pythia-410m pilot through the whole pipeline; inspect logs and fix.
6. Fact-check and red-team review → `reports/reviews/`.
7. Run stages in order: **1a (simple top-3-PCA plots and observations) first**, then 1b, 2, 3 (causal/behavioral), 4 (ablations), 5 (sensitivity), 6 (more models), moving from simple to complex. After each stage, update the results, paper, README and site, then push to GitHub and HF.
8. Final triple-check, fact-check and red-team review; final paper and site.

Waiting on long runs follows the CLAUDE.md cadence: check every 5, 15 or 30 minutes depending on run length. Compute is light (extraction takes minutes to about an hour per model), so most time goes to engineering and analysis. Estimated total: about 3–5 days of wall-clock work.

## Verification

- `uv run pytest` passes, covering all statistical and geometric ground-truth tests above.
- The end-to-end pilot `uv run python -m phylogeo.run --config configs/pilot.yaml` on Pythia-410m runs every phase; logs are clean and outputs make sense.
- Stage 1a: the top-3-PCA plots render for all layers. The days-of-week calibration shows a circle. Visual check in the in-app browser.
- Sanity checks:
  - The synthetic Brownian-tree cloud gives TA≈1 and δ≈0.
  - Random vectors give TA≈1/3.
  - Days and months recover a circle (positive control).
  - Random-init Qwen and Pythia step0 are ≈ the lexical baselines.
- `--phases report` regenerates every table and figure from `results/` alone, with no hand-typed numbers.
- Visual checks of the site and paper in the in-app browser. GitHub Pages deploy succeeds. HF collection pages render.

## Key risks

- **Name ambiguity or tokenization** could dominate the signal. Mitigated by the polysemy filter, head-noun-disjoint triplets, and both readouts.
- **Qwen3.5 hybrid architecture.** Gated DeltaNet carries a recurrent state, so left-padding could leak into it, and fast kernels may be missing on Windows. Mitigated by the batched-vs-unbatched equality test (with a fallback to batch size 1, which is cheap at this scale) and by the Qwen3-1.7B standard-attention replication.
- **A 2B model may know less biology than larger models.** Scale trends from Qwen3.5-0.8B→2B and from the Pythia suite let us state this limitation quantitatively.
- **TimeTree API** could be unavailable or incomplete. Primary analyses need topology only, which is covered by Open Tree and NCBI.
- **Low within-class signal** (a negative result) is acceptable and will be reported as such. The preregistered falsification criteria apply.
