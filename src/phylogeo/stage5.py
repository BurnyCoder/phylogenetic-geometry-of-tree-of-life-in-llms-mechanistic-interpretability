"""Stage 5: more models — scale (Qwen3.5-0.8B/2B, Qwen3-1.7B, Pythia 70m-2.8b) and training
dynamics (Pythia-1.4B checkpoints), on the same species, triplets and metric.

Global context: a light per-run analysis (no recomputation of controls) gives
TA_within at every layer, the max-T permutation p (within-stratum shuffles,
same triplets as stage 1b), the family cross-fitted TA_within and the Mantel
correlation at the cross-fitted layer. ``summarize`` collects all runs into
results/stage5/summary.json and two figures: best-layer TA_within vs parameter
count and vs training step (log scale), with the untrained and lexical
baselines from stage 1b as reference lines.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from phylogeo.config import RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.stage5")

PARAMS = {  # parameter counts (from the model cards; Pythia names are nominal sizes)
    "Qwen/Qwen3.5-0.8B": 0.85e9, "Qwen/Qwen3.5-2B": 2.27e9, "Qwen/Qwen3-1.7B": 1.7e9,
    "EleutherAI/pythia-70m": 70e6, "EleutherAI/pythia-160m": 160e6, "EleutherAI/pythia-410m": 410e6,
    "EleutherAI/pythia-1b": 1.0e9, "EleutherAI/pythia-1.4b": 1.4e9, "EleutherAI/pythia-2.8b": 2.8e9,
}


def run_scale(cfg: RunConfig, load_species, layer_distances, out_dir, write_json) -> dict:
    from phylogeo.analysis import _rng
    from phylogeo.stats.mantel import mantel_r
    from phylogeo.stats.triplets import crossfit_ta, distinct_mask, maxT_ta, sample_triplets, within_group_mask

    df, tree = load_species()
    strata = df["stratum"].to_numpy()
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    trip = sample_triplets(tree, cfg.k_per_node, _rng(cfg, "triplets"))
    within = within_group_mask(trip, strata) & distinct_mask(trip, df["head_noun"].to_numpy())
    D, _ = layer_distances(cfg.name)
    obs, p, _ = maxT_ta(D, trip, within, strata, int(cfg.extra.get("scale_n_perm", 199)), _rng(cfg, "maxT"))
    cf, layers = crossfit_ta(D, trip, within, fam, 10, _rng(cfg, "cf"))
    ml = int(np.bincount(layers).argmax())
    step = re.search(r"step(\d+)", cfg.revision or "")
    res = {"run": cfg.name, "model_id": cfg.model_id, "revision": cfg.revision,
           "params": PARAMS.get(cfg.model_id), "step": int(step.group(1)) if step else None,
           "n_layers": int(D.shape[0] - 1), "ta_within_by_layer": obs, "ta_within_max": float(np.nanmax(obs)),
           "best_layer_rel_depth": float(np.nanargmax(obs) / (D.shape[0] - 1)), "maxT_p": p,
           "crossfit_ta_within": cf, "modal_layer": ml, "mantel_pathlen_modal": float(mantel_r(D[ml], tree.path_length()))}
    write_json(out_dir(cfg.name, "stage5") / "scale.json", res)
    log.info("%s: TA_within max %.3f, cross-fit %.3f, max-T p %.3f", cfg.name, res["ta_within_max"], cf, p)
    return res


def summarize(write_json) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*/stage5/scale.json"))]
    s1b = json.loads((RESULTS_DIR / "qwen35_2b" / "stage1b" / "results.json").read_text())
    ref = {k: s1b["point_ta_within"][k] for k in ("qwen35_2b_random", "pythia410m_step0", "char3", "glove")
           if k in s1b["point_ta_within"]}
    out = {"runs": runs, "reference_ta_within": ref}
    (RESULTS_DIR / "stage5").mkdir(exist_ok=True)
    write_json(RESULTS_DIR / "stage5" / "summary.json", out)
    final = [r for r in runs if r["step"] is None or r["step"] == 143000]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for fam, mk, col in (("Qwen", "o", "#2a78d6"), ("pythia", "s", "#eb6834")):
        rs = sorted([r for r in final if fam.lower() in r["model_id"].lower()], key=lambda r: r["params"] or 0)
        ax.plot([r["params"] for r in rs], [r["crossfit_ta_within"] for r in rs], marker=mk, color=col, label=fam)
        for r in rs:
            ax.annotate(r["model_id"].split("/")[1], (r["params"], r["crossfit_ta_within"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    for name, v in ref.items():
        ax.axhline(v, ls="--", lw=0.8, color="0.5")
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 6e7, v, f" {name}", fontsize=6, va="bottom", color="0.4")
    ax.axhline(1 / 3, ls=":", lw=0.8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("cross-fitted TA_within")
    ax.set_title("Within-stratum triplet accuracy vs model size", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stage5" / "scale.png", dpi=150)
    plt.close(fig)
    ck = sorted([r for r in runs if r["step"] is not None and "1.4b" in r["model_id"]], key=lambda r: r["step"])
    if ck:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot([max(r["step"], 0.5) for r in ck], [r["crossfit_ta_within"] for r in ck], marker="o", color="#eb6834")
        ax.axhline(1 / 3, ls=":", lw=0.8, color="0.3")
        ax.set_xscale("log")
        ax.set_xlabel("training step (step 0 plotted at 0.5)")
        ax.set_ylabel("cross-fitted TA_within")
        ax.set_title("Pythia-1.4B: within-stratum triplet accuracy over training", fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "stage5" / "training_dynamics.png", dpi=150)
        plt.close(fig)
    return out
