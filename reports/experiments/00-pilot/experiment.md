# Experiment 00: pilot on Pythia-410m (pipeline debugging)

**Status:** done, 2026-11-20. The pilot was declared in the preregistration; it is used only to debug the pipeline, and its results do not change any hypothesis.

## What was run
- **Model:** [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m), final checkpoint, bf16.
- **Inputs:** raw text with BOS prepended; 24 layers, giving 25 hidden-state arrays including the embeddings.
- **Name sets:** 400 species, the days of the week, the months, and the 79 conflict-triplet names, each in 8 templates. Extraction took about 2 minutes.
- **Command:**
  ```powershell
  uv run python -m phylogeo.run --config configs/pilot.yaml --phases extract,pca,analyze,conflict
  ```
- **Settings:** reduced permutation counts (199), bootstrap (200) and cross-fitting splits (5), for speed.

## What it was for
1. Check that name tokens are located correctly. The logs show, e.g., `<|endoftext|>The blue whale is an animal.` → name tokens `['Ġblue', 'Ġwhale']`.
2. Check that every phase runs and writes results: `results/pilot_pythia410m/`.
3. Check that the figures render (top-3 PCA panels, tree overlay, layer grids, 3D HTML).

## Results
All numbers are in `results/pilot_pythia410m/` and rendered in [`results/RESULTS.md`](../../../results/RESULTS.md). Qualitatively:
- **Stage-1a checks C1, C2, C3 and C3b pass at every layer, including layer 0.** Layer 0 is the token embedding of the last name token (the head noun), so these checks are lexically confounded (see [learnings](learnings.md)).
- **TA_within** (head-noun-disjoint, within stratum) sits just around the preregistered 0.40 "meaningful" threshold. Its CI is above chance, and it beats the character-3-gram and input-embedding baselines in paired bootstrap differences.
- **Conflict triplets:** the pilot representation chooses the tree-closer partner in a minority of triplets, similar to the lexical baseline. This is a first hint of folk/lexical geometry.

## Pipeline changes made after the pilot
- Added the exploratory C3b check (within-stratum shuffle), logged as deviation 10.
- No changes to the primary analysis.
