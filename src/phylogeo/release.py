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
    from phylogeo.verification.cards import model_card

    card = model_card(conf)
    (out / "README.md").write_text(card, encoding="utf-8")
    log.info("release probe written to %s (fit r = %.3f)", out, fit_r)


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
    from phylogeo.verification.cards import activation_card
    api.upload_file(path_or_fileobj=activation_card().encode(), path_in_repo="README.md", repo_id=act_id, repo_type="dataset")
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
