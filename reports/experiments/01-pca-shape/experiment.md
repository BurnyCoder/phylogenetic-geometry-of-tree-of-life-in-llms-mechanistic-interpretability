# Experiment 01 (stage 1a): do the species look like a tree in the top-3 PCA directions?

**Hypothesis H1a** (preregistered, exploratory): in some layer's top-3 PCs, species form a branching tree: clade arms with nested sub-branches, and a tree overlay shorter than the shuffled-leaf null.

## Setup
- **Model:** Qwen3.5-2B (revision `15852e8`), bf16, chat template, batch size 1.
- **Inputs:** 400 species × 8 templates.
- **Readout:** residual stream at the last name token, layers 0–24.
- **Species vectors:** per-template mean-centering over species, then the average over templates. PCA is fitted per layer on the 400 species vectors.
- **Calibration, plotted with the same code:**
  - days of the week and months (a circle is expected);
  - random-init Qwen3.5-2B (same species);
  - Brownian motion simulated on the reference tree.
- **Checks C1–C4 and C3b:** 1,000 permutations each (see the preregistration).
- **Command:**
  ```powershell
  uv run python -m phylogeo.run --config configs/qwen35_2b.yaml --phases pca
  ```
- **Outputs:** `results/qwen35_2b/stage1a/`. Interactive 3D: `pca3d.html` (colored, tree overlay, layer slider) and `pca3d_uncolored.html`.
- **Animated view (added 2026-11-20, display only).** The site's 3-D player and `reports/figures/viz/pca3d.gif` show the same top-3 PCs at every layer. Consecutive layers are rotated onto each other by orthogonal Procrustes, so axes do not flip, and each layer is scaled to unit RMS radius. This changes no statistic. Data: `results/qwen35_2b/viz/pca3d.json` (phase `viz_data`).

## Anti-bias procedure
1. Look only at the uncolored grids (`grid_uncolored_pc12.png`, `grid_uncolored_pc23.png`) and describe the shape. **Description 1** below is written before any colored plot is opened.
2. Then open the colored grids, the 2D panels with the tree overlay, and the zoom-ins. **Description 2** below.
3. Then read the C1–C4 / C3b numbers. The table is auto-generated in [`results/RESULTS.md`](../../../results/RESULTS.md).

## Description 1 (uncolored, written before seeing clades)
*Written 2026-11-20 08:36 from `grid_uncolored_pc12.png` and `grid_uncolored_pc23.png` only.*

- **Low-dimensional structure is weak.** Each of PC1–PC3 explains only about 3–8% of the variance at every layer (PC1+PC2+PC3 is about 15–20%). There is no dominant 2–3-d manifold like the days-of-week circle; most structure is outside the top 3 PCs.
- **Layers 0–1.** Layer 0 (token embeddings) is a 1-D band, a dense end tapering to a sparse tail, with 1–2 extreme outliers. Layer 1 is a dense core plus 3–4 small, very tight satellite clusters far from it. Tight satellites this early look lexical: groups of names sharing a token.
- **Layers 2–5: the most "branching" appearance.** Layers 4–5 show a dense centre with 3–4 arms or lobes radiating out. In layer 5 (PC1–2) one arm runs lower-left, one upper-left and a lobe sits on the right, so it looks star-like.
- **Layers 6–14.** A diffuse, roughly isotropic blob with a few small, tight peripheral clusters (e.g. top-left in layers 6–9, right edge in layers 9–12). No clear arms.
- **Layers 15–24.** More structured. In PC1–2, layers 20–23 form a curved V/U-shaped band (an arch at layer 21), i.e. a continuous gradient bent between two extremes, plus 1–2 tiny tight clusters. In PC2–3 these layers show a central mass with one arm hanging downward (a "T/Y" shape).
- **Overall impression before seeing labels.** Not an obvious tree. It is a diffuse cloud with a few protruding arms, a handful of isolated micro-clusters, and in late layers a curved band. Labels are needed to tell whether the arms are clades (tree-like), coarse classes (clusters) or a gradient.

## Description 2 (colored, pre-fix species set)
*From `results/qwen35_2b_prefix/stage1a/grid_colored_pc12.png` and `grid_colored_pc23.png`, viewed 2026-11-20 08:37, before the implementation review led to re-freezing the species set.*

- **Caveat.** That palette used two near-identical blues for mammals and ray-finned fish. Fish were told apart from mammals mainly by position, which is a weakness fixed in the re-run.
- **Layers 4–5: the "arms" are coarse classes.** Each class sits on its own lobe or arm radiating from the centre:
  - birds on one arm;
  - fish (ray-finned and cartilaginous) on another;
  - mammals in a central lobe;
  - insects on a third arm, with arachnids as a tight cluster at its tip;
  - molluscs, echinoderms and cnidarians on the opposite side.
- **Layers 6–19.** The classes occupy overlapping regions. Arachnids, and in layers 16–18 cartilaginous fish, form tight satellite clusters at the edges.
- **Layers 20–23: the curved band is a class gradient.** In PC1–PC2 it runs birds → mammals → reptiles/amphibians (with molluscs and echinoderms) → ray-finned and cartilaginous fish. Insects hang below the middle, and arachnids sit in a tight cluster at the bottom.
  - This ordering is **not** the phylogenetic one: birds are closer to reptiles than to mammals.
  - It looks more like an ecological air → land → water gradient. That is an **exploratory, post-hoc hypothesis** to be tested later with the conflict triplets (whales, bats, penguins) and habitat covariates. No claim is made here.
- **No visible sub-branching inside classes in the top 3 PCs.** Birds, for example, form one spread cloud without sub-arms. Any within-class tree must live in other dimensions, which stage 1b tests.

## Re-run on the re-frozen species set
After the review fixes (deviations 11–18), the species set changed: names were fixed, and *Hippopotamus*, *Dugong* and others were added. The pre-fix plots had already been seen, so the anti-bias step cannot be repeated blind for the re-run. The re-run is compared against descriptions 1–2 above.

## Re-run results (re-frozen set, `results/qwen35_2b/stage1a/`)
All numbers are in the auto-generated table in [`results/RESULTS.md`](../../../results/RESULTS.md) → Run `qwen35_2b` → Stage 1a, and in `checks.json`. The qualitative shape is unchanged from descriptions 1–2.

**Calibration (positive and negative controls)**
- **Days of the week and months form clean circles in calendar order** (PC1–PC2, layer 12; `calib_days_layer12_pc12.png`, `calib_months_layer12_pc12.png`). This reproduces Engels et al. in Qwen3.5-2B, so the pipeline recovers known low-dimensional geometry.
- **The synthetic Brownian-motion-on-the-tree cloud** (`calib_synthetic_brownian_tree.png`) shows long, clean arms per clade in the top 3 PCs. That is what a dominant tree signal looks like. The real species cloud looks very different.
- **Random-init Qwen** (`calib_random_init_grid.png`; column "C1 random-init"):
  - At layer 0 its class purity equals the trained model's. Layer-0 structure is therefore pure token identity: shared head nouns get the same embedding even with random weights.
  - In deeper layers it falls to near chance, while the trained model's purity rises sharply.

**Spotlight small multiples** (`spotlight_layer21_pc12.png`). At layer 21 the curved band orders the strata:
- **One end:** cartilaginous fish, cnidarians, echinoderms, molluscs, crustaceans; ray-finned fish and the "other" group (lampreys, lungfish, coelacanths) on a lower arm.
- **Middle:** amphibians, reptiles, mammals.
- **Other end:** insects and birds.

Birds and insects are phylogenetically distant but are both flyers. The dominant top-PC axis therefore looks like a habitat / locomotion gradient (water → land → air), not phylogeny. This is a **post-hoc, exploratory observation**. It is turned into a testable hypothesis in stage 4: within mammals, marine species should sit toward the aquatic end and bats toward the aerial end. The test will use EltonTraits foraging strata, on a held-out species sample, before any claim is made.

**Zoom-in, mammals, layer 21** (`zoom_mammals_layer21.png`):
- Rodents and lagomorphs cluster together; this is Glires, a real clade.
- Primates cluster.
- A tight artiodactyl group sits at the far end of PC1; to be identified, probably the cetaceans.
- So some order-level structure is present (C2 mammals above its null and above the layer-0 lexical value).

## Reading of H1a (preregistered rule + shape judgment)
- **Formal checks.** C1, C2 and C3 beat their permutation nulls in every layer from 2 to 24, so the preregistered rule labels these layers "tree-consistent (checks 1–3 pass)". In layers 0–1, C2 for birds fails.
- **Effect sizes against the lexical reference (layer 0) and the random-init model:**
  - Class purity (C1) roughly doubles from layer 0 to layers 4–5 and 20–21.
  - Order purity *within birds* (C2) rises from a non-significant value at layer 0 to clearly above its null in middle and late layers.
  - Order-level structure within the coarse classes is therefore not just head-noun identity.
- **Branching beyond the coarse classes in the top 3 PCs (C3b, exploratory)** is significant but **no larger than at layer 0**: the overlay/within-stratum-null ratio is about the same as the lexical reference. In the top 3 PCs, the within-class tree signal is not stronger than what word identity alone produces.
- **Shape by eye:**
  - Layers 4–5: a class-level star (arms = coarse classes radiating from a centre).
  - Late layers: a curved, ecology-like gradient through the classes.
  - **No visible nested branching inside classes**, unlike the synthetic tree.
- **Verdict (exploratory):** H1a is **partially supported**. The top-3 PCs show class-level arms and some order-level clustering, beyond the lexical and random-init references. The overall top-3 shape is closer to "clusters + ecological gradient" (alternatives b/c) than to the nested tree of the Brownian calibration. As preregistered, this does not decide the question for higher dimensions; stage 1b (triplet accuracy over all 2,048 dimensions) does.
