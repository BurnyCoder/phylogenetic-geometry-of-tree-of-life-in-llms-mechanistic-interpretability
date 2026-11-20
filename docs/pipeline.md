# Pipeline

There is one entry point:

```powershell
uv run python -m phylogeo.run --config configs/<run>.yaml --phases <a,b,c> [--name-sets <set,...>]
```

Phases run in the given order. Each logs to `logs/<timestamp>_<run>.log` and to the terminal, and writes JSON and figures to `results/<run>/<stage>/`. Large arrays go to `DATA_DIR` (set in `.env`), never to git.

## Configs (`configs/*.yaml`)

| Field | Meaning |
|---|---|
| `name` | Run name, used for the `results/` and activation folders |
| `model_id`, `revision` | Hugging Face model and **pinned** revision (commit hash or Pythia `stepN`) |
| `random_init` | `true`: same architecture with freshly initialized weights (seeded control) |
| `chat_template` | Wrap prompts in the model's chat template (user turn, thinking disabled) |
| `prepend_bos` | Prepend BOS (Pythia omits it by default) |
| `seed` | Base seed. Every analysis component derives its own stream: `phylogeo.analysis._rng` |
| `n_perm`, `n_boot`, `k_per_node` | Permutations, bootstrap draws, triplets per tree node |
| `extra.controls` | Runs used as controls in stage 1b |
| `extra.random_init_run` | The architecture-matched untrained control |

## Phases

| Phase | Module | Needs | Writes |
|---|---|---|---|
| `species` | `data/species.py` | internet (Wikidata QLever, Open Tree, NCBI) | `data/species.csv`, `data/species_tree.nwk`, `data/conflict*.csv`, `data/FREEZE.json` |
| `dates` | `data/tree.py` | TimeTree API | `DATA_DIR/timetree/species_node_ages.npy` (not redistributed) |
| `extract` | `extract.py` | GPU, model download | `DATA_DIR/activations/<run>/<name_set>/{last,mean}.npy`, `names.txt` |
| `pca` | `viz/pca.py`, `viz/plots.py` | activations | `results/<run>/stage1a/` (2D/3D plots, `checks.json`) |
| `power` | `simulate.py` | tree | `results/power/` |
| `analyze` | `analysis.py` | main and control activations, dates | `results/<run>/stage1b/results.json` |
| `conflict` | `analysis.py` | conflict activations | `results/<run>/conflict/conflict.json` |
| `stage2` | `stage2.py`, `probes/flat.py`, `treeness.py` | activations | `results/<run>/stage2/` |
| `stage3` / `stage3_causal` | `stage3.py`, `causal.py`, `behavior.py` | GPU | `results/<run>/stage3/results.json` |
| `stage4` | `stage4.py` | extra extractions (`species_sci`, raw format) | `results/<run>/stage4/` |
| `eco` | `eco.py`, `data/heldout.py` | `heldout_eco` activations | `results/<run>/eco/` |
| `scale`, `scale_summary` | `stage5.py` | activations of each stage-5 run | `results/<run>/stage5/scale.json`, `results/stage5/` |
| `stage6` | `stage6.py` | main and control activations | `results/<run>/stage6/` |
| `report` | `report.py` | results JSON | `results/RESULTS.md`, `results/<run>/SUMMARY.md`, README and summary blocks |
| `viz_data` | `viz/viz_data.py` | activations (CPU) | `results/<run>/viz/` (tree-ordered distance percentiles, 2-D PCA incl. the Brownian-tree expectation, nearest neighbours) |
| `site` | `site.py`, `viz/figures.py` | results, `results/<run>/viz/` | `site/` (deployed by `.github/workflows/pages.yml`), `reports/figures/viz/*.png`, README *Visual summary* gallery |
| `paper` | `paper.py` | results, `typst` | `reports/paper.typ`, `reports/paper.pdf` |

## Activation arrays

`last.npy` and `mean.npy` are float32 arrays of shape `[n_names, n_templates, n_layers + 1, d_model]`:
- index 0 on the layer axis is the embedding output;
- index `l` is the output of decoder layer `l`.

`last` is the residual stream at the final token of the name. `mean` is the average over the name's tokens. The `names.txt` file lists the rows. Every analysis checks it against the frozen name list before use (`phases.check_names`).

## Name sets

| Name set | Content |
|---|---|
| `species` | The 400 frozen species (common names) |
| `species_sci` | The same species as scientific binomials (stage 4) |
| `species_polysemous` | Polysemous names excluded from the main set (stage 4) |
| `conflict` | Animals of the 36 conflict triplets |
| `heldout_eco` | 159 held-out mammals and birds (Addendum A) |
| `days`, `months` | Calibration sets (expected circles) |

## Reproducibility notes
- **Batch size 1.** Batched bf16 forward passes differ from unbatched ones by up to 2.5% in hidden-state norm for this hybrid-attention model (preregistration deviation 7; [learnings](../reports/experiments/learnings.md)).
- **One GPU job at a time on Windows.** Two GPU processes can exhaust the commit limit (OS error 1455). Background jobs are chained.
- **Generated numbers only.** Every number in `results/RESULTS.md`, the README headline block, `reports/summary.md` and the paper is produced by `report` or `paper` from the JSON files.
