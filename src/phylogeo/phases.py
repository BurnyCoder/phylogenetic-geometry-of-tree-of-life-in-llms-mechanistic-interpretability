"""Pipeline phases called by ``phylogeo.run`` (one function per phase).

Global context: each ``phase_<name>(cfg)`` reads frozen inputs (data/species.csv,
data/species_tree.nwk), activations from ``DATA_DIR/activations/<run>/<name_set>``
and writes JSON/CSV/PNG/HTML to ``results/<run>/``. The analysis code follows the
preregistration (reports/preregistration.md); section numbers are cited inline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from phylogeo.config import ACT_DIR, DATA_DIR, DATA_REPO_DIR, RESULTS_DIR, RunConfig
from phylogeo.data.tree import RefTree

log = logging.getLogger("phylogeo.phases")
RANK_ORDER = ["phylum", "class", "order", "family", "genus"]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def load_species(name: str = "species") -> tuple[pd.DataFrame, RefTree]:
    """Frozen species table and its reference tree (leaves in table order)."""
    df = pd.read_csv(DATA_REPO_DIR / f"{name}.csv", keep_default_na=False)
    nwk = (DATA_REPO_DIR / f"{name}_tree.nwk").read_text(encoding="utf-8").strip()
    tree = RefTree(nwk, [f"ott{i}" for i in df["ott_id"]])
    return df, tree


def names_and_templates(name_set: str) -> tuple[list[str], list[str]]:
    """Names and templates for a name set (species / species_polysemous / conflict / days / months)."""
    from phylogeo.prompts import CALIB_TEMPLATES, DAYS, MONTHS, TEMPLATES

    if name_set == "days":
        return DAYS, CALIB_TEMPLATES
    if name_set == "months":
        return MONTHS, CALIB_TEMPLATES
    df = pd.read_csv(DATA_REPO_DIR / f"{name_set}.csv", keep_default_na=False)
    return df["common_name"].tolist(), TEMPLATES


def act_dir(run: str, name_set: str) -> Path:
    return ACT_DIR / run / name_set


def out_dir(run: str, sub: str = "") -> Path:
    p = RESULTS_DIR / run / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")


def layer_distances(run: str, name_set: str = "species", readout: str = "last", variant: str = "centered"):
    """[L, n, n] cosine distances for a run (variants: centered | raw | zscore)."""
    from phylogeo.extract import load_activations
    from phylogeo.geometry import cosine_distance, species_vectors, zscore_dims

    X = load_activations(act_dir(run, name_set), readout)
    V = species_vectors(X, center=(variant != "raw"))
    if variant == "zscore":
        V = zscore_dims(V)
    return cosine_distance(V), X


# ---------------------------------------------------------------------------
# dates: TimeTree ages for the frozen tree (cached locally, never committed)
# ---------------------------------------------------------------------------

def phase_dates(cfg: RunConfig) -> None:
    from phylogeo.data.tree import date_tree

    df, tree = load_species()
    ages = date_tree(tree, df["ncbi"].astype(int).tolist())
    p = DATA_DIR / "timetree"
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "species_node_ages.npy", ages)
    log.info("saved node ages to %s (not redistributed: TimeTree terms)", p)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def phase_extract(cfg: RunConfig) -> None:
    """Extract activations for every name set in ``cfg.extra['name_sets']``."""
    import torch

    from phylogeo.extract import extract_names, load_model, save_activations

    tok, model = load_model(cfg.model_id, cfg.revision, cfg.random_init, cfg.dtype, cfg.seed)
    for ns in cfg.extra.get("name_sets", ["species"]):
        names, templates = names_and_templates(ns)
        log.info("extracting %s: %d names x %d templates", ns, len(names), len(templates))
        last, mean = extract_names(tok, model, names, templates, chat=cfg.chat_template,
                                   prepend_bos=cfg.prepend_bos)
        save_activations(act_dir(cfg.name, ns), last, mean, names)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# stage 1a: top-3 PCA + H1a checks (preregistration section 3, stage 1a)
# ---------------------------------------------------------------------------

def _name_token_counts(model_id: str, names: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return np.array([len(tok(" " + n, add_special_tokens=False)["input_ids"]) for n in names])


def phase_pca(cfg: RunConfig) -> None:
    from phylogeo.extract import load_activations
    from phylogeo.geometry import species_vectors
    from phylogeo.simulate import brownian_tree
    from phylogeo.viz.pca import classify_h1a, h1a_checks, knn_purity, pca_top3
    from phylogeo.viz.plots import circle_png, pca_2d_png, pca_3d_html, pca_layer_grid_png

    rng = np.random.default_rng(cfg.seed)
    n_perm = int(cfg.extra.get("pca_n_perm", 1000))
    df, tree = load_species()
    names = df["common_name"].tolist()
    strata = df["stratum"].to_numpy()
    od = out_dir(cfg.name, "stage1a")
    X = load_activations(act_dir(cfg.name, "species"), "last")
    V = species_vectors(X)  # [L, n, d]
    L = V.shape[0]
    coords, evrs, tcoords = [], [], []
    for l in range(L):
        c, ev, p = pca_top3(V[l])
        coords.append(c)
        evrs.append(ev)
        Xt = X[:, :, l] - X[:, :, l].mean(0, keepdims=True)  # per-template centering
        tcoords.append(p.transform(Xt.reshape(-1, Xt.shape[-1])))
    title = f"{cfg.model_id} ({cfg.name}): {len(names)} animal species"
    # Anti-bias step: uncolored views first (no clades, no tree).
    pca_layer_grid_png(coords, evrs, strata, od / "grid_uncolored_pc12.png", title + " — uncolored", colored=False)
    pca_layer_grid_png(coords, evrs, strata, od / "grid_uncolored_pc23.png", title + " — uncolored", colored=False,
                       dims=(1, 2))
    pca_3d_html(coords, evrs, names, strata, None, od / "pca3d_uncolored.html", title + " (uncolored)", colored=False)
    # Colored views with tree overlay and single-prompt points.
    pca_layer_grid_png(coords, evrs, strata, od / "grid_colored_pc12.png", title)
    pca_layer_grid_png(coords, evrs, strata, od / "grid_colored_pc23.png", title, dims=(1, 2))
    pca_3d_html(coords, evrs, names, strata, tree, od / "pca3d.html", title, template_coords_by_layer=tcoords)
    # Checks per layer.
    confounds = {
        "log_sitelinks": np.log(df["sitelinks"].to_numpy(float)),
        "n_name_tokens": _name_token_counts(cfg.model_id, names),
        "name_length": df["common_name"].str.len().to_numpy(),
        "n_words": df["n_words"].to_numpy(),
    }
    ctrl_V = None
    ctrl = cfg.extra.get("random_init_run")
    if ctrl and (act_dir(ctrl, "species") / "last.npy").exists():
        ctrl_V = species_vectors(load_activations(act_dir(ctrl, "species"), "last"))
    checks = []
    for l in range(L):
        ch = h1a_checks(V[l], tree, strata, df["order"].to_numpy(), confounds, n_perm, rng)
        ch["layer"] = l
        if ctrl_V is not None:
            ch["C1_purity_random_init"] = knn_purity(pca_top3(ctrl_V[l])[0], strata)
        ch["reading"] = classify_h1a(ch)
        log.info("layer %d: C1 purity %.3f (null %.3f, p=%.3g) | C2 %s | C3 len %.1f vs null %.1f (p=%.3g) | "
                 "C3b within-stratum null %.1f (p=%.3g) -> %s",
                 l, ch["C1_purity"], ch["C1_purity_null"], ch["C1_p"],
                 {k: f"{v['purity']:.2f}/null {v['null']:.2f}/p {v['p']:.3f}"
                  for k, v in ch["C2_within_stratum_order_purity"].items()},
                 ch["C3_overlay_length"], ch["C3_null_mean"], ch["C3_p"],
                 ch["C3b_within_stratum_null_mean"], ch["C3b_p"], ch["reading"])
        checks.append(ch)
    write_json(od / "checks.json", checks)
    # 2D panels for the layer with the strongest tree-overlay effect and the middle/last layers.
    best = int(np.argmin([c["C3_overlay_length"] / c["C3_null_mean"] for c in checks]))
    for l in sorted({best, L // 2, L - 1}):
        pca_2d_png(coords[l], evrs[l], strata, tree, od / f"pca2d_layer{l}.png", f"{title} — layer {l}")
        pca_2d_png(coords[l], evrs[l], strata, None, od / f"pca2d_layer{l}_uncolored.png",
                   f"{title} — layer {l} (uncolored)", colored=False)
    # Zoom-ins: PCA within single strata, colored by order (top 8 orders).
    for s in ("mammals", "birds", "insects"):
        m = strata == s
        sub_df = df[m].reset_index(drop=True)
        top = sub_df["order"].value_counts().index[:8]
        lab = np.where(sub_df["order"].isin(top), sub_df["order"], "other")
        c, ev, _ = pca_top3(V[best][m])
        _orders_png(c, ev, lab, od / f"zoom_{s}_layer{best}.png", f"{s} only, layer {best}, colored by order")
    # Calibration: days / months circles, random-init, synthetic Brownian tree.
    for ns, labels in (("days", None), ("months", None)):
        p = act_dir(cfg.name, ns)
        if (p / "last.npy").exists():
            Xc = load_activations(p, "last")
            Vc = species_vectors(Xc)
            lab = (p / "names.txt").read_text(encoding="utf-8").split()
            for l in sorted({L // 2, best, L - 1}):
                c, ev, _ = pca_top3(Vc[l])
                circle_png(c, ev, lab, od / f"calib_{ns}_layer{l}_pc12.png", f"{ns}, layer {l}: PC1-PC2")
                circle_png(c, ev, lab, od / f"calib_{ns}_layer{l}_pc23.png", f"{ns}, layer {l}: PC2-PC3", dims=(1, 2))
    if ctrl_V is not None:
        rc = [pca_top3(ctrl_V[l]) for l in range(L)]
        pca_layer_grid_png([r[0] for r in rc], [r[1] for r in rc], strata, od / "calib_random_init_grid.png",
                           f"random-init {cfg.model_id}: same species")
    Xs = brownian_tree(tree, V.shape[-1], rng)
    c, ev, _ = pca_top3(Xs - Xs.mean(0))
    pca_2d_png(c, ev, strata, tree, od / "calib_synthetic_brownian_tree.png",
               "synthetic Brownian motion on the reference tree (what a true tree looks like)")
    log.info("stage 1a done; best overlay layer %d", best)


def _orders_png(coords, evr, labels, path, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for g in pd.unique(labels):
        m = labels == g
        ax.scatter(coords[m, 0], coords[m, 1], s=14, label=g)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.legend(fontsize=7); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# power simulation (preregistration section 4.5)
# ---------------------------------------------------------------------------

def phase_power(cfg: RunConfig) -> None:
    from phylogeo.simulate import power_curve

    df, tree = load_species()
    signals = cfg.extra.get("signals", [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    res = power_curve(tree, df["stratum"].to_numpy(), df["head_noun"].to_numpy(), signals,
                      n_sims=int(cfg.extra.get("n_sims", 40)), n_perm=int(cfg.extra.get("power_n_perm", 199)),
                      k_per_node=cfg.k_per_node, dim=int(cfg.extra.get("dim", 256)), seed=cfg.seed)
    od = out_dir("power")
    write_json(od / "power.json", res)
    ok = [r for r in res if r["power"] >= 0.8]
    mde = min(ok, key=lambda r: r["signal"])["ta_within_mean"] if ok else None
    write_json(od / "summary.json", {"min_detectable_ta_within_80pct_power": mde, "curve": res})
    log.info("minimum detectable TA_within at 80%% power: %s", mde)


# ---------------------------------------------------------------------------
# stage 1b and conflict triplets (implemented in analysis.py)
# ---------------------------------------------------------------------------

def load_conflict():
    """Conflict-triplet names and their Open Tree tree (unique species leaves)."""
    df = pd.read_csv(DATA_REPO_DIR / "conflict.csv", keep_default_na=False)
    nwk = (DATA_REPO_DIR / "conflict_tree.nwk").read_text(encoding="utf-8").strip()
    leaves = sorted({int(o) for o in df["ott_id"] if int(o) > 0})
    return df, RefTree(nwk, [f"ott{i}" for i in leaves])


def phase_analyze(cfg: RunConfig) -> None:
    from phylogeo.analysis import run_stage1b

    run_stage1b(cfg, load_species, layer_distances, act_dir, out_dir, write_json)


def phase_conflict(cfg: RunConfig) -> None:
    from phylogeo.analysis import run_conflict

    run_conflict(cfg, load_conflict, layer_distances, act_dir, out_dir, write_json)


def phase_report(cfg: RunConfig) -> None:
    """Regenerate results/RESULTS.md from all results JSON files."""
    from phylogeo.report import write_results_md

    runs = sorted(p.name for p in RESULTS_DIR.iterdir() if (p / "stage1a").exists() or (p / "stage1b").exists())
    log.info("report written to %s", write_results_md(runs))


def phase_stage2(cfg: RunConfig) -> None:
    """Stage 2: geodesic vs angular, delta-hyperbolicity, NJ, flat-subspace probe."""
    from phylogeo.stage2 import run_stage2

    run_stage2(cfg, load_species, layer_distances, act_dir, out_dir, write_json)
