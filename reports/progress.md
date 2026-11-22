# Progress

## 2026-11-22: tree order inside groups, in 3-D (post hoc visualization)

- **Done.** The H1 highlight in the README now has an animated 3-D figure (also an interactive card on the site with a group dropdown). It shows every group with ≥ 20 H1 decision nodes (birds, mammals, ray-finned fish, insects) in three columns:
  - a perfect tree code (Brownian motion on the reference tree);
  - Qwen3.5-2B at the H1 layer;
  - the same network with random weights.
- **Method.**
  - Every panel uses unit-length vectors, centred on the group, then top-3 PCA. Unit length makes the picture approximate the cosine geometry H1 scores; raw-norm PCA made random weights look tree-like.
  - The reference subtree is fitted into each cloud with least-squares internal nodes (squared-change parsimony, equal to Brownian-motion ancestral states; the phylomorphospace approach).
- **Numbers per panel.** Triplet accuracy in 3-D and in all dimensions on the H1 triplets, tree fit vs species shuffles, and per-group paired family-block bootstrap CIs of trained minus random. All are generated into the README table from `results/qwen35_2b/viz/within3d.json`.
- **Caveat found.**
  - In the 3-D view the trained model beats random weights in birds and mammals.
  - In all dimensions no single group's CI excludes zero; only the pooled H1 test does.
  - The README states this next to the figure.

## 2026-11-22: independent audit of results, methods and claims

- **Done.** Every claim was traced to code and results. The headline numbers were re-derived from stored activations with independent code (new phase `audit`; no model inference or training), and every citation was re-checked on the web. Report: [reviews/2026-11-22-independent-audit.md](reviews/2026-11-22-independent-audit.md).
- **Reproduced.** TA_within over all qualifying triplets, without the per-node cap, matches the sampled values; the max-T, Holm, rank-p, McNemar, binomial and exact Mann–Whitney values re-derive.
- **Corrected.**
  - Random-init Qwen is not above chance by its own max-T test.
  - Untrained networks are at chance on triplets whose names share no character 3-gram.
  - On the same held-out pairs, raw distances keep more within-stratum order than the flat probe.
  - Two reference titles, author details and the Qwen parameter counts; the novelty claim now separates phylogeny from lexical taxonomies.
- **New caveat.** The H1 margin over the untrained networks excludes zero with the registered family-level bootstrap, but not with orders as blocks. H1 stays supported as registered.
- **Regenerated.** Tables, site, figures and paper.
- **Status.** The planned work is complete. Remaining items are follow-ups listed in the README to-do: order-level resampling registered for any replication, a co-occurrence-neutral analysis, larger models.


## 2026-11-20 18:10: 3-D view now trained vs random init, side by side

- **Player.** Trained Qwen3.5-2B (left) and random-init Qwen3.5-2B (right) are shown at the same layer, with one clock and linked cameras.
- **Fair comparison.** Each layer is scaled by its median species radius, and both panels share a per-layer box. The untrained network's outlier names outside the box are counted in its title.
- **README.** The GIF is now side by side.

## 2026-11-20 17:50: animated 3-D view of the top-3 principal components

- **The player.** The site's player shows every layer:
  - play and pause, a layer scrubber with smooth interpolation, and speed;
  - views of Qwen3.5-2B, the untrained Qwen3.5-2B, or the expected Brownian-tree shape;
  - a tree overlay and auto-rotation.
- **README.** An animated GIF of the same view leads the *Visual summary*.
- **Method (display only).** Consecutive layers are aligned by orthogonal Procrustes, so no statistic changes.

## 2026-11-20 17:30: interactive figures in the README and on the site

- **17 labelled figures**, each with a verdict computed from the results files:
  - *ground truth vs observed:* taxonomy sunburst, distance heatmaps in tree order, PCA against the Brownian-tree expectation;
  - *positive:* H1 forest, layer profile, neighbour relatedness, scale and training dynamics, sensitivity;
  - *mixed:* per-stratum, depth profile, causal, habitat;
  - *negative:* secondary vs untrained, conflict triplets, flat probe, treeness;
  - *output examples:* a neighbour grid.
- **Where they appear.** The site shows them interactively, plus an explorer of the nearest species for all 400 species. The README shows PNGs in a new *Visual summary* section near the top, each linking to its interactive card.
- **Pipeline.** A new `viz_data` phase derives compact data from the activations on the CPU.

## 2026-11-20 16:55: all stages complete; stage-3 causal rerun finished

**Done since 12:20**
- **Stage 5 (more models).** The effect grows with scale (Pythia 70M → 2.8B, Qwen3.5 0.8B → 2B), is not specific to the hybrid architecture (Qwen3-1.7B is similar), and is learned during training (Pythia-1.4B checkpoints: chance until step 512, then a jump by step 1000). See [experiment 07](experiments/07-more-models-stage5/experiment.md).
- **Final review round 1.** All must-fix findings were addressed ([response](reviews/2026-11-20-final-review-round1-response.md)).
- **Stage 3 causal rerun (deviations 19–20).**
  - The run used 99 random subspaces and a class-only subspace control.
  - C-swap is supported after Holm correction, with a small effect. C-abl is not supported.
  - The class-only subspace does as well as the phylogeny probe on both tests, so no phylogeny-specific causal role is shown. See [experiment 04](experiments/04-causal-behavior-stage3/experiment.md).
  - The numbers are in the generated `results/RESULTS.md`.
- **Release.**
  - Activations of all 24 runs and the probe (with the class-only caveat on its card) are on Hugging Face, in the project collection.
  - The site is deployed by GitHub Pages.
  - The paper, README and summary are regenerated with the causal results.

**Now**
- Final fact-check of the causal write-up, then merge to main (which redeploys the site).

**Not done (listed in the README to-do as possible extensions)**
- R3-steer (geodesic steering) and R3-sae (no SAE for post-trained Qwen3.5-2B).
- EltonTraits ecology/body-size baseline, S5 within-stratum bins, power simulation with ≥ 200 simulations per cell.
- A within-class causal readout.

**Overall:** the planned stages 1a–6 are done; about 95% of the plan is complete. What remains is the final fact-check and red-team pass and the merge. No GPU job is running.

## 2026-11-20 12:20: stages 1–4 and 6 complete; stage 3 causal rerun and stage 5 running

**Done since 09:30**
- **Stage 2 (exploratory).** The flat probe generalizes to held-out families far above the shuffled-tree and GloVe controls. There is no geodesic advantage, δ_rel lies between the class-blob and Gaussian nulls, and the neighbor-joining tree is closer to the reference than random trees.
- **Stage 3.**
  - Behavior: the model's relatedness judgments are slightly above chance and agree modestly with its representation.
  - Causal: the first run had too few random subspaces to ever pass Holm, and a mismatched off-target baseline. Both were fixed (deviation 19, committed before the rerun); the rerun is queued after stage 5.
- **Stage 4 (ablations).** The effect is robust to normalization, prompt format and polysemous names. Latin binomials carry no structure. Poincaré is not better than flat. The probe captures within-stratum order.
- **Addendum A (habitat axis, held-out species).** The aquatic end is confirmed; the aerial end is not.
- **Stage 6 (sensitivity).** The model beats the best control on every triplet seed and every 80% species subsample. Templates barely matter, and the cross-fit layer and probe are stable across seeds.
- **Reporting.**
  - Report tables for every stage, with a generated headline block in the README and `reports/summary.md`.
  - The site covers all stages, with an interactive per-layer chart; GitHub Pages is enabled and deploys on merge to main.
  - The paper was rewritten with full sections and references.
  - `docs/pipeline.md`.
- **Hugging Face.** The all-species flat probe is uploaded, and activations of the finished runs are uploading.

**Now (background, one GPU job at a time)**
- **Stage 5.** Finished so far: Qwen3.5-0.8B, Qwen3-1.7B, Pythia 70m/160m/410m/1b. Now running Pythia-1.4b, then 2.8b, then 12 Pythia-1.4B checkpoints, whose downloads are still finishing.
- **Stage 3 causal rerun.** 99 random subspaces for ablation and swap, starting after stage 5. Estimated about 3 hours.

**Next**
- Stage-5 summary and experiment write-up.
- Stage-3 causal results and experiment 04.
- Final fact-check and NeurIPS-style red-team review.
- Merge to main, which deploys the site, then upload the remaining activations.

**Overall:** about 85% done. Remaining wall-clock is dominated by GPU jobs: stage 5 needs about 1.5–2 hours and the stage-3 rerun about 3 hours, checked every 30 minutes.

## 2026-11-20 09:30: stage 1 complete; stage 2 running

**Done**
- **Literature.** Review plus an independent fact-check.
- **Preregistration and review.** Preregistration pushed before data. An independent implementation review found 9 serious issues, all fixed before any confirmatory statistic; deviations 11–18 are logged.
- **Data.** Species set re-frozen: 400 species, NCBI-first ranks, UTF-8 names, and the genus-title rule removed. TimeTree re-dated 308 of 390 internal nodes.
- **Power simulation.** Run on the frozen tree before extraction. Minimum detectable TA_within at 80% power is below the 0.40 smallest effect of interest.
- **Extraction.** Qwen3.5-2B, random-init Qwen3.5-2B, Pythia-410m, Pythia-410m step0.
- **Stage 1a (exploratory).**
  - Class-level arms and within-class order clustering beyond the lexical and random-init references.
  - Days and months form circles (positive control).
  - No nested within-class tree in the top 3 PCs.
  - A late-layer, ecology-like gradient (water → land → air).
- **Stage 1b (confirmatory).**
  - H1 is **supported**, with a meaningful effect: TA_within beats every required control.
  - S1 and S2 are supported.
  - S3 and S4 are not: no evidence beyond spelling or co-occurrence.
  - S5 is not: no divergence-time effect beyond topology.
  - Conflict triplets follow folk categories.
  - Tables: `results/qwen35_2b/SUMMARY.md`.
- **Hugging Face.** Dataset and collection created.

**Now**
- Stage 2: geodesic vs angular, δ-hyperbolicity, neighbor-joining, the flat-subspace probe with controls (running).

**Next**
- Stage 3: causal ablation and interchange of the probe subspace, plus behavioral judgments.
- Stage 4: ablations, including the post-hoc habitat-gradient hypothesis on held-out species.
- Stage 5: more models (Qwen3.5-0.8B, Qwen3-1.7B, Pythia scale and checkpoints).
- Stage 6: sensitivity.
- Then the paper, the site and the final reviews.

**Overall:** about 50% done. Remaining estimate: about 1–2 days, mostly stages 3–6 and the paper.
