"""Addendum A: test of the habitat-gradient hypothesis (H_eco) on held-out species.

Global context: the "habitat axis" is PC1 of the layer-21 species vectors of the
400 primary species (Qwen3.5-2B), oriented so birds are positive and ray-finned
fish negative. Held-out mammals and birds (data/heldout_eco.csv) are centred with
the primary set's per-template means and projected onto that fixed axis. Then
(one-sided, Holm across four registered slots):
  M1  marine mammals (ForStrat M) < ground mammals (G)        Mann-Whitney
  M2  aerial mammals (A) > ground mammals (G)                  Mann-Whitney (only if n_A >= 5)
  B1  Spearman(axis, % water foraging) < 0                     birds
  B2  Spearman(axis, % aerial foraging) > 0                    birds
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from phylogeo.config import DATA_REPO_DIR, RunConfig

log = logging.getLogger("phylogeo.eco")


def habitat_axis(X_primary: np.ndarray, strata: np.ndarray, layer: int):
    """Return (template means [T, d], unit axis [d]) defining the oriented layer-`layer` PC1."""
    from sklearn.decomposition import PCA

    Xl = X_primary[:, :, layer]  # [n, T, d]
    mu_t = Xl.mean(0)  # per-template mean over primary species
    V = (Xl - mu_t).mean(1)
    pc = PCA(n_components=1).fit(V).components_[0]
    s = V @ pc
    if s[strata == "birds"].mean() < s[strata == "ray-finned fish"].mean():
        pc = -pc
    return mu_t, pc


def run_eco(cfg: RunConfig, act_dir, out_dir, write_json, layer: int = 21) -> dict:
    from phylogeo.analysis import holm
    from phylogeo.extract import load_activations
    from phylogeo.phases import check_names, load_species

    df, _ = load_species()
    check_names(cfg.name, "species")
    check_names(cfg.name, "heldout_eco")
    Xp = load_activations(act_dir(cfg.name, "species"), "last")
    mu_t, axis = habitat_axis(Xp, df["stratum"].to_numpy(), layer)
    del Xp
    h = pd.read_csv(DATA_REPO_DIR / "heldout_eco.csv", keep_default_na=False)
    Xh = load_activations(act_dir(cfg.name, "heldout_eco"), "last")[:, :, layer]
    score = ((Xh - mu_t).mean(1)) @ axis
    h["axis_score"] = score
    res = {"layer": layer, "n_mammals": int((h.stratum == "mammals").sum()), "n_birds": int((h.stratum == "birds").sum()),
           "tests": {}}
    mam = h[h.stratum == "mammals"]
    g = mam.loc[mam.forstrat_mammal == "G", "axis_score"].to_numpy()
    for key, code, alt in (("M1_marine_lt_ground", "M", "less"), ("M2_aerial_gt_ground", "A", "greater")):
        x = mam.loc[mam.forstrat_mammal == code, "axis_score"].to_numpy()
        if len(x) >= 5:
            t = mannwhitneyu(x, g, alternative=alt)
            res["tests"][key] = {"n": int(len(x)), "n_ground": int(len(g)), "median": float(np.median(x)),
                                 "median_ground": float(np.median(g)), "p": float(t.pvalue)}
        else:
            res["tests"][key] = {"n": int(len(x)), "evaluable": False,
                                 "median": float(np.median(x)) if len(x) else None}
    b = h[h.stratum == "birds"]
    w = spearmanr(b["axis_score"], b["water_pct_bird"].astype(float))
    a = spearmanr(b["axis_score"], b["aerial_pct_bird"].astype(float))
    res["tests"]["B1_water_negative"] = {"rho": float(w.statistic), "p": float(w.pvalue / 2 if w.statistic < 0 else 1 - w.pvalue / 2)}
    res["tests"]["B2_aerial_positive"] = {"rho": float(a.statistic), "p": float(a.pvalue / 2 if a.statistic > 0 else 1 - a.pvalue / 2),
                                          "n_nonzero_aerial": int((b["aerial_pct_bird"].astype(float) > 0).sum())}
    # Keep every registered slot in the multiplicity denominator, including the empty bat group.
    ev = {k: v.get("p", 1.) for k, v in res["tests"].items()}
    for k, v in holm(ev).items():
        if "p" in res["tests"][k]:
            res["tests"][k]["holm_p"] = v
    # Descriptive: the primary 400 by stratum on the same axis, and per-species held-out scores.
    res["heldout_scores"] = h[["common_name", "stratum", "forstrat_mammal", "water_pct_bird", "aerial_pct_bird",
                               "axis_score"]].to_dict(orient="records")
    write_json(out_dir(cfg.name, "eco") / "results.json", res)
    log.info("H_eco: %s", {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in res["tests"].items()})
    return res
