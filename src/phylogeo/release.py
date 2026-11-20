"""Hugging Face release: final flat probe weights and the activation arrays.

Global context: the user's workflow puts models and data into one Hugging Face
collection. ``build_release`` trains the stage-2 flat probe (k = 10, angular,
seed 0) on *all* 400 species at the stage-1b layer and writes it as safetensors
with a generated model card; held-out performance in the card comes from the
stage-2 and stage-6 results JSON (the released probe itself has no held-out
species). ``upload_release`` pushes the probe repo and the activation dataset
(``upload_folder`` splits commits and resumes in huggingface_hub 2.x) and adds both to the collection.
Docs: https://huggingface.co/docs/huggingface_hub/guides/upload
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from phylogeo.config import DATA_DIR, RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.release")
PROBE_REPO = "phylogeo-flat-probe-qwen3.5-2b"
ACT_REPO = "phylogenetic-geometry-activations"
RELEASE_DIR = DATA_DIR / "release"


def build_release(cfg: RunConfig, load_species, layer_distances, act_dir) -> None:
    import torch
    from safetensors.torch import save_file
    from sklearn.decomposition import PCA

    from phylogeo.extract import load_activations
    from phylogeo.geometry import species_vectors
    from phylogeo.probes.flat import probe_distances, train_probe

    df, tree = load_species()
    s1b = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())
    s2 = json.loads((RESULTS_DIR / cfg.name / "stage2" / "results.json").read_text())
    s6p = RESULTS_DIR / cfg.name / "stage6" / "results.json"
    L = int(s1b["H1"]["modal_layer"])
    V = species_vectors(load_activations(act_dir(cfg.name, "species"), "last"))[L]
    P = tree.path_length()
    target_scale = float(np.pi / P.max())  # tree distance -> [0, pi], as in cv_probe
    pca = PCA(n_components=100, random_state=0).fit(V)
    Z = pca.transform(V)
    z_scale = float(Z.std())
    m = train_probe(Z / z_scale, P * target_scale, k=10, steps=2000, seed=0)
    Dh = probe_distances(m, Z / z_scale)
    iu = np.triu_indices(len(V), 1)
    fit_r = float(np.corrcoef(Dh[iu], P[iu])[0, 1])
    out = RELEASE_DIR / "probe"
    out.mkdir(parents=True, exist_ok=True)
    save_file({"W": m.W.detach().contiguous(), "b": m.b.detach().contiguous(), "log_beta": m.log_beta.detach().reshape(1),
               "pca_components": torch.tensor(np.ascontiguousarray(pca.components_), dtype=torch.float32).contiguous(),
               "pca_mean": torch.tensor(np.ascontiguousarray(pca.mean_), dtype=torch.float32).contiguous()}, str(out / "probe.safetensors"))
    conf = {"model_id": cfg.model_id, "revision": cfg.revision, "layer": L, "readout": "last name token",
            "k": 10, "metric": "angular", "z_scale": z_scale, "target_scale": target_scale,
            "per_template_centering": True, "n_species": len(V), "fit_pearson_all_species": fit_r}
    (out / "config.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    with torch.no_grad():
        coords = m.project(torch.tensor(Z / z_scale, dtype=torch.float32)).numpy()
    df_c = df[["common_name", "sci_name", "stratum", "ncbi", "ott_id"]].copy()
    for i in range(coords.shape[1]):
        df_c[f"z{i}"] = coords[:, i]
    df_c.to_csv(out / "species_coordinates.csv", index=False)
    pm = s2["probe"]
    held = f"{pm['model']['test_test_pearson'][0]:.3f} ± {pm['model']['test_test_pearson'][1]:.3f}"
    sens = ""
    if s6p.exists():
        s6 = json.loads(s6p.read_text())["probe_seed"]["test_test_pearson"]
        sens = f" Over {len(s6['values'])} further seeds: {s6['mean']:.3f} ± {s6['sd']:.3f}."
    cb = pm.get("class_blob")
    cb_txt = (f" The same probe trained on a class-only input (each species at its stratum centroid plus noise) reaches "
              f"held-out r = {cb['test_test_pearson'][0]:.3f}." if cb else "")
    card = f"""---
license: apache-2.0
base_model: {cfg.model_id}
tags: [interpretability, probing, phylogeny, representation-geometry]
---
# Flat-subspace phylogeny probe for {cfg.model_id}

A 10-dimensional linear "flat subspace" probe in the style of [Goodfire's Evo 2 analysis](https://www.goodfire.com/research/phylogeny-manifold),
trained to map Qwen3.5-2B residual-stream vectors of animal species names to distances on the Open Tree of Life.

- **Input:** for each species, the residual stream after decoder layer {L} at the last token of the name. Take the mean over 8 chat-formatted prompts, after per-template mean-centering over the 400 species.
- **Map:**
  - `x_pca = (x - pca_mean) @ pca_components.T / z_scale`
  - `z = (x_pca - b) @ W.T`
  - `d = exp(log_beta) * arccos(cos(z_i, z_j)) / target_scale`
- **Training:** all 400 species, Adam, 2000 full-batch steps, seed 0. Pearson r with tree path length on the training species is {fit_r:.3f}.
- **Generalization** (from the paper's cross-validation, not this checkpoint): held-out-family Pearson r {held} (5-fold family holdout, 5 seeds).{sens} A shuffled-tree control probe reaches {pm['shuffled_tree']['test_test_pearson'][0]:.3f}.
- **Files:**
  - `probe.safetensors`: weights and PCA;
  - `config.json`;
  - `species_coordinates.csv`: the 10-d coordinates of the 400 species.

Code, preregistration and paper: https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability

**Limitations.**
- The probe is fit to path length in the Open Tree topology; it is not a model of divergence times.
- It was trained on English common names; Latin binomials carry no comparable structure.
- **The held-out correlation mainly reflects coarse groups.**{cb_txt} Within groups the probe adds little over the raw representation distances (see the paper).
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    log.info("release probe written to %s (fit r = %.3f)", out, fit_r)


ACT_CARD = """---
license: apache-2.0
tags: [interpretability, activations, phylogeny, representation-geometry]
---
# Residual-stream activations for animal species names

Activations used in *Phylogenetic geometry of the tree of life in LLMs* (code and paper:
https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability).

Layout: `<run>/<name_set>/{last,mean}.npy` and `names.txt`.

**Arrays.** float32, shape `[n_names, n_templates, n_layers + 1, d_model]`.
- Layer index 0 is the embedding output; index `l` is the output of decoder layer `l`.
- `last` is the residual stream at the final token of the name; `mean` is the mean over the name's tokens.

**Runs.** Model, pinned revision and prompt format for each run are in the repository's `configs/<run>.yaml`:
- the main model and its untrained control: `qwen35_2b`, `qwen35_2b_random`;
- the raw-text variant `qwen35_2b_raw`;
- the Pythia controls: `pythia410m`, `pythia410m_step0`;
- the stage-5 models, including the Pythia-1.4B training checkpoints `pythia14b_step*`.

**Name sets.** `species` (400 frozen species; order in `names.txt` and in `data/species.csv` of the repository), `species_sci`,
`species_polysemous`, `conflict`, `heldout_eco`, `days`, `months`.

**Sources and licences.**
- Species names come from Wikidata (CC0).
- The models are Qwen3.5 and Qwen3 (Apache-2.0) and Pythia (Apache-2.0).
- No TimeTree data are included.
"""


def upload_release(cfg: RunConfig) -> None:
    from pathlib import Path

    from huggingface_hub import HfApi, get_token

    ns = os.environ.get("HF_NAMESPACE", "BurnyCoder")
    coll = os.environ.get("HF_COLLECTION")
    # HF_HOME points to the D: cache, but `hf auth login` stores the token under the default
    # ~/.cache/huggingface; fall back to it (the token is never logged or written anywhere).
    default = Path.home() / ".cache" / "huggingface" / "token"
    api = HfApi(token=get_token() or (default.read_text().strip() if default.exists() else None))
    probe_id, act_id = f"{ns}/{PROBE_REPO}", f"{ns}/{ACT_REPO}"
    api.create_repo(probe_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=probe_id, folder_path=str(RELEASE_DIR / "probe"), repo_type="model",
                      commit_message="Flat-subspace phylogeny probe (generated release)")
    log.info("uploaded probe to %s", probe_id)
    api.create_repo(act_id, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=ACT_CARD.encode(), path_in_repo="README.md", repo_id=act_id, repo_type="dataset")
    runs = [r for r in os.environ.get("UPLOAD_RUNS", "").split(",") if r]  # optional: only finished runs
    # huggingface_hub 2.x: upload_folder splits large folders into several commits and resumes when re-run.
    api.upload_folder(repo_id=act_id, folder_path=str(DATA_DIR / "activations"), repo_type="dataset",
                      allow_patterns=[f"{r}/**" for r in runs] or None, ignore_patterns=["pilot_*/**", "pilot_*"],
                      commit_message=f"Activations: {', '.join(runs) or 'all runs'}")
    log.info("uploaded activations to %s", act_id)
    if coll:
        for item, kind in ((probe_id, "model"), (act_id, "dataset")):
            try:
                api.add_collection_item(coll, item_id=item, item_type=kind, exists_ok=True)
            except Exception as e:  # already present or permission issue: report, do not fail the upload
                log.warning("collection add %s: %s", item, e)
