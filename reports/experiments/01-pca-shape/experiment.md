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

## Checks and reading
*(pending)*
