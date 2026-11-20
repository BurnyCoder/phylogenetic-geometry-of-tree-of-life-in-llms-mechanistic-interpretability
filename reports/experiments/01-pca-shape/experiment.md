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

## Description 2 (colored, tree overlay, zoom-ins)
*(pending)*

## Checks and reading
*(pending)*
