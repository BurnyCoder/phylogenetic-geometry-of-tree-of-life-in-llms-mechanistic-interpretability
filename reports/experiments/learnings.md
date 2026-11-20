# Learnings (all experiments)

Running log of what was learned from the literature, from errors, and from the experiments: what was tried, what went wrong, why, and how it was fixed. Experiment-specific learnings are in each `reports/experiments/<id>/learnings.md`.

## From the literature review
- **Goodfire's Evo 2 phylogeny result reports no controls** ([post](https://www.goodfire.com/research/phylogeny-manifold)): no permutation test, no random-init model, no shuffled tree. An independent reproduction got much weaker numbers (geodesic r 0.38 vs 0.78). *Consequence:* every geometric metric here is computed against an untrained model, lexical and co-occurrence baselines, and class-blob and Brownian-tree simulations, with permutation p-values.
- **Global correlations are dominated by coarse classes.** With 13 strata, a pure "folk class" oracle already correlates strongly with tree distance. *Consequence:* the primary metric is *within-stratum* triplet accuracy (TA_within), not a global Mantel r.
- **Head nouns leak phylogeny** (phylotext-mammals: unmasking head nouns raised r from ~0 to 0.19). *Consequence:* the primary metric uses only triplets whose three head nouns differ.
- **Co-occurrence statistics alone produce hierarchical geometry** ([Nava & Wyart 2026](https://arxiv.org/abs/2605.23821)). *Consequence:* a GloVe baseline and co-occurrence-discordant triplets (S4).
- **Whitening manufactures orthogonality** (critique of [Park et al.](https://arxiv.org/abs/2406.01506)). *Consequence:* no whitening in any primary analysis.

## From engineering errors
| Problem | Cause | Fix | Why this fix |
|---|---|---|---|
| Wikidata SPARQL query for ~20k species timed out (60 s) | The public WDQS scans all species items for the sitelinks filter | Use the [QLever](https://qlever.dev/api/wikidata) endpoint: the same query returns in 1–16 s | Same data (Wikidata), faster engine; responses cached under `DATA_DIR/cache` |
| `ete4` failed to build (Cython error) on Windows / Python 3.12 | No wheel, compile failure in `ete4/core/operations.pyx` | Take ranks from the Open Tree Taxonomy lineage (`taxonomy/taxon_info`) | One fewer dependency. OTT is the taxonomy of the reference tree itself, so ranks and topology are consistent |
| OTT "Reptilia" flagged barren/inconsistent | Reptilia is paraphyletic without birds; OTT does not maintain it | Reptiles = Testudines ∪ Lepidosauria ∪ Crocodylia | Same extant non-avian reptiles |
| Unit test "tree metric gives TA = 1" failed with edge-count path length | Edge counts are not ultrametric: a shallow outgroup can be fewer edges away than a deep sister | Added `RefTree.lca_depth_distance()` (max depth − depth(LCA)) as the triplet-consistent oracle; logged as deviation 8 | Mathematical fact; S1 kept as registered |
| Geodesic on a circle with one large gap gave low correlation with arc length | The smallest connected k leaves the largest gap unbridged, so the kNN graph is an open chain ("tearing") | Documented with a unit test. Geodesics are compared with angular distance and with the class-blob null, not assumed correct | Goodfire's and Modell's method shares this limitation |
| `hf download` of Qwen3.5-2B failed after the weights downloaded | `WinError 1314`: creating symlinks requires a privilege the client lacks | `huggingface_hub.snapshot_download` from the project venv completed the snapshot (copy fallback) | No admin rights or Developer Mode needed |
| Batched and unbatched hidden states differ by up to 2.5% relative norm (Qwen3.5-2B, bf16) | Batch-size-dependent bf16 kernels, not padding (both padding sides and the unpadded prompt showed the same error) | Batch size 1 for every prompt | Exact and reproducible; still fast (3.2k prompts in minutes) |
| Common names miscapitalized ("eurasian magpie", "Guinea pig") | Wikipedia titles always capitalize the first word; Wikidata labels are inconsistent (birds use IOC Title Case) | WordNet-based rule: the first word keeps its capital only if WordNet lists it only in capitalized form, plus a compass/ocean rule and a possessive rule | Deterministic and tested. 6 compound words remain capitalized ("Megamouth shark", "Redback spider", "Bluntnose sixgill shark", "Beadlet anemone", "Snakelocks anemone", "Elkhorn coral") and 2 genus-like names ("Phylloxera", "Rosalia longicorn") |
| "beluga" in the fish stratum was the sturgeon | Title "Beluga (sturgeon)" with its parenthetical stripped | Titles that needed a parenthetical qualifier → polysemy flag (deviation 4) | Wikipedia's own judgment that the bare name is ambiguous |
| Conflict-set names failed to place: dog, little brown bat, silverfish | Dog is a subspecies in OTT; *Myotis lucifugus* is a broken taxon; silverfish's scientific name was misspelled | Subspecies → species; the bat placed at its congener *Myotis myotis*; spelling fixed (deviation 6) | Triplet answers are unchanged (all splits are above genus) |
| Heredocs with nested quotes broke in Git Bash | Shell quoting | Write files with the editor tool instead | — |
| Hook-based residual edits did not show up in `output_hidden_states` (unit test on a tiny random Qwen3.5) | transformers 5 records `hidden_states` with its own forward hook registered before ours, so it stores the *pre-edit* layer output, although the edited output does flow to the next layer | Tests read the edited value with a second hook registered after the edit, and check that layer L+1 changes. Causal effects are measured on logits, never on the reported `hidden_states` of the edited layer | Avoids silently "verifying" an intervention that the recorded states do not reflect |
| A probe script run outside the package wrote a config to the C: Hugging Face cache | `.env` (HF_HOME) is loaded only when `phylogeo.config` is imported, and some modules import transformers first | `phylogeo/__init__.py` now imports `config`, so `.env` is loaded on any package import | Models must never land on the nearly full C: drive |
| trackio dashboard showed "No projects" while probes were training | trackio stores runs under `HF_HOME/trackio`. The training process loaded `.env` (HF_HOME on D:), but `uv run trackio show` did not | Launch the dashboard with the same `HF_HOME` (README) | Both sides read the same database |
| Running extraction, power simulation and TimeTree dating in parallel failed with "paging file is too small" (os error 1455) and a 1.5 MiB `ArrayMemoryError` | The Windows commit limit is 50.6 GB (32 GB RAM + a 19.8 GB pagefile on the almost-full C: drive, which cannot grow); other apps already commit about 34 GB; loading a 4.5 GB safetensors file needs commit headroom | Run GPU jobs **sequentially**. The user's paging-file settings and running apps (WSL, browsers) were left untouched | Changing system settings is the user's decision. Sequential runs cost only minutes |

## From the independent implementation review
- **Simple nuisance variables can pass a sophisticated test.**
  - A 1-D "representation" holding only the number of name tokens, or only log Wikipedia popularity, reaches within-stratum triplet accuracy ≈ 0.38–0.40 with p < 0.02.
  - The reason is that naming conventions ("X-crested Y-warbler") and popularity are clustered by family.
  - *Lesson:* every metric needs controls for the cheapest *non-semantic* explanations, not only for the obvious lexical ones. Both are now required H1 baselines.
- **Global centering + cosine can manufacture structure.** If the spread of points depends on a covariate that is clustered by clade (popularity), centering on the global mean turns norm heterogeneity into apparent within-group structure (review simulation: TA 0.37 with zero phylogeny). *Lesson:* report a stratum-centered variant.
- **A decision rule must fail loudly.** H1 could be declared "supported" when a control's activations were missing. *Lesson:* decision functions take the full required set and return "not evaluable" otherwise.
- **Taxonomy databases have holes.** OTT leaves Primates as "no rank"; NCBI lacks classes for turtles and crocodilians and has merged taxids. *Lesson:* use NCBI first, resolve merged ids, fall back to OTT, and test known cases (chimpanzee and siamang share an order).
- **Imputation can create impossible data.** Giving undated tree nodes their child's age produced 0-Mya phylum-level splits. *Lesson:* leave missing values missing (NaN) and exclude them.
- **Time and topology are nearly collinear within rank bins** (|ρ| up to 0.99). *Lesson:* control for topology (partial correlation) before claiming the model encodes divergence *time*.
- **HTTP decoding defaults bite.** `requests` decodes text without a charset as ISO-8859-1, so "Grévy's" became "GrÃ©vy's" in a frozen dataset. *Lesson:* force UTF-8 for SPARQL CSV, and grep frozen data for mojibake.
- **Plots: never two blues for two classes.** Two near-identical blues hid the mammal/fish distinction. *Lesson:* validate palettes with a script (dataviz `validate_palette.js`); for >3 categories in a scatter, use composite color × marker encoding and spotlight small multiples.

## From the pilot (Pythia-410m)
- **The stage-1a checks C1, C2, C3 and C3b all pass at layer 0 (token embeddings).** At layer 0 the last-name-token vector is the embedding of the head noun ("shark", "crab", "frog"). Species sharing a head noun are usually related, so lexical identity alone produces class purity, order purity and a short tree overlay. *Consequence:* the stage-1a checks cannot separate lexical from phylogenetic structure, and layer 0 must be reported as the lexical reference for every stage-1a effect size. The decisive test is H1, whose triplets are head-noun-disjoint. C3b (within-stratum shuffle) was added after the pilot to separate "tree" from "clusters" (deviation 10).

## From stages 3–6 and the release
- **A permutation-style test against n random controls cannot give p < 1/(n+1).**
  - Stage 3 first compared the probe subspace with 20 and 10 random subspaces, so the smallest possible p-values were 0.048 and 0.091.
  - The registered Holm family of 2 needs p ≤ 0.025, so the test could never succeed whatever the data.
  - *Lesson:* before running a rank test, compute its p-value floor against the corrected threshold.
  - Fixed before the rerun, with 99 subspaces (deviation 19).
- **Compare like with like in off-target checks.**
  - The no-edit loss was token-weighted over 20 sentences, while the edited losses were unweighted means over the first 10. That apparent effect was an artifact.
  - *Lesson:* compute baseline and intervention with the same function, and pass the edit as an argument (a unit test now checks that an identity edit reproduces the baseline exactly).
- **Recompute controls on the same perturbed draw** (stage 6). Comparing marginal spreads overstates uncertainty about the model-vs-control gap; the paired margin is the quantity of interest.
- **The prompt barely matters with centering and a last-token readout.** Even a single template reproduces the main TA_within (stage 6). Species composition is the main source of variance, which is why the family-block bootstrap is the right CI.
- **Hugging Face token location.**
  - With `HF_HOME` moved to D:, `huggingface_hub` looks for the token under `HF_HOME`, but `hf auth login` had stored it in `~/.cache/huggingface/token`, which caused a 401 on upload.
  - The release code falls back to the default token path. The token is never logged or copied.
- **API drift.** `HfApi.upload_large_folder` no longer exists in huggingface_hub 2.4; `upload_folder` now splits large folders into several commits and resumes when re-run.
- **Generated text blocks keep docs honest.** The README and summary headline tables are injected by the `report` phase between markers, so no hand-typed number can go stale.
