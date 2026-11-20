"""Held-out mammals and birds with EltonTraits foraging strata (preregistration Addendum A).

Global context: the habitat-gradient hypothesis (H_eco) came from looking at the
400 primary species, so it is tested on NEW species. They come from the same
Wikidata candidate pool and rules as the primary set: common-name title, OTT
match, leaf of the synthetic tree, not polysemous. They must not be in the
primary or polysemous sets, and must match EltonTraits 1.0 (Wilman et al. 2014,
https://esapubs.org/archive/ecol/E095/178/; MD5-verified) by scientific name.
The top 80 mammals and top 80 birds by sitelinks are kept.
"""

from __future__ import annotations

import logging

import pandas as pd

from phylogeo.config import DATA_DIR, DATA_REPO_DIR

log = logging.getLogger("phylogeo.heldout")
ELTON_DIR = DATA_DIR / "eltontraits"
MD5 = {"MamFuncDat.txt": "59c3eee29d3ed0a33a002975a5a8cc75", "BirdFuncDat.txt": "d6197b2cd90ca3ece0a7393abbf8b7fc"}


def load_elton() -> tuple[pd.DataFrame, pd.DataFrame]:
    """EltonTraits mammal and bird tables (downloaded once from Ecological Archives, MD5-checked)."""
    import hashlib

    import requests

    ELTON_DIR.mkdir(parents=True, exist_ok=True)
    for f, md5 in MD5.items():
        p = ELTON_DIR / f
        if not p.exists():
            r = requests.get(f"https://esapubs.org/archive/ecol/E095/178/{f}", timeout=120,
                             headers={"User-Agent": "phylogeo-research/0.1 (academic research script)"})
            r.raise_for_status()
            p.write_bytes(r.content)
        if hashlib.md5(p.read_bytes()).hexdigest() != md5:
            raise ValueError(f"MD5 mismatch for {f}")
    mam = pd.read_csv(ELTON_DIR / "MamFuncDat.txt", sep="\t", encoding="latin-1")
    bird = pd.read_csv(ELTON_DIR / "BirdFuncDat.txt", sep="\t", encoding="latin-1")
    return mam, bird


def build_heldout_eco(n_per_class: int = 80, min_sitelinks: int = 5, prefilter_factor: int = 6) -> pd.DataFrame:
    """Select and freeze ``data/heldout_eco.csv`` (+ tree) following Addendum A."""
    from phylogeo.data.opentree import induced_subtree
    from phylogeo.data.species import (QUOTAS, _capitalized_words, annotate_with_ott, common_name, ensure_wordnet,
                                       freeze, head_noun, polysemy_flag, query_candidates)
    from phylogeo.data.tree import RefTree

    ensure_wordnet()
    cand = query_candidates(min_sitelinks)
    proper = _capitalized_words(cand["title"].tolist())
    cand["common_name"] = [common_name(t, s, proper) for t, s in zip(cand["title"], cand["sci_name"])]
    cand = cand[cand["common_name"].notna()].copy()
    cand = cand[~cand["common_name"].str.lower().duplicated(keep=False)]
    cand["qualified_title"] = cand["title"].str.contains(r"\(", regex=True)
    # Same prefilter as the primary build (cached API calls), mammals and birds only.
    parts = [cand[cand["wd_stratum"] == s].nlargest(QUOTAS[s] * prefilter_factor, "sitelinks") for s in ("mammals", "birds")]
    ann = annotate_with_ott(pd.concat(parts).drop_duplicates("wikidata"))
    ann = ann[ann["stratum"].isin(["mammals", "birds"])]
    _, dropped = induced_subtree(ann["ott_id"].tolist())
    ann = ann[~ann["ott_id"].isin(dropped)].copy()
    ann["polysemous"] = [polysemy_flag(n) or bool(q) for n, q in zip(ann["common_name"], ann["qualified_title"])]
    used = set(pd.read_csv(DATA_REPO_DIR / "species.csv")["ott_id"]) | \
        set(pd.read_csv(DATA_REPO_DIR / "species_polysemous.csv")["ott_id"])
    ann = ann[~ann["polysemous"] & ~ann["ott_id"].isin(used)]
    mam, bird = load_elton()
    mam_fs = dict(zip(mam["Scientific"], mam["ForStrat-Value"]))
    bird["water_pct"] = bird["ForStrat-watbelowsurf"] + bird["ForStrat-wataroundsurf"]
    bw = dict(zip(bird["Scientific"], bird["water_pct"]))
    ba = dict(zip(bird["Scientific"], bird["ForStrat-aerial"]))
    rows = []
    for s, lookup in (("mammals", mam_fs), ("birds", bw)):
        sub = ann[(ann["stratum"] == s) & ann["sci_name"].isin(lookup)].nlargest(n_per_class, "sitelinks")
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["forstrat_mammal"] = out["sci_name"].map(mam_fs).where(out["stratum"] == "mammals", "")
    out["water_pct_bird"] = out["sci_name"].map(bw).where(out["stratum"] == "birds")
    out["aerial_pct_bird"] = out["sci_name"].map(ba).where(out["stratum"] == "birds")
    out["n_words"] = out["common_name"].str.split().str.len()
    out["head_noun"] = out["common_name"].map(head_noun)
    cols = ["common_name", "sci_name", "wikidata", "ncbi", "ott_id", "sitelinks", "stratum", "phylum", "class",
            "order", "family", "genus", "n_words", "head_noun", "forstrat_mammal", "water_pct_bird", "aerial_pct_bird"]
    out = out[cols].reset_index(drop=True)
    nwk, dr = induced_subtree(out["ott_id"].tolist())
    assert not dr
    # A selected taxon can be an ancestor of another (it then appears as an internal
    # node, not a leaf); drop such taxa and rebuild the tree.
    import re

    leaves = set(re.findall(r"[(,](ott\d+)(?=[,)])", nwk))
    keep = out["ott_id"].map(lambda i: f"ott{i}" in leaves)
    if (~keep).any():
        log.info("dropping %d held-out taxa that are internal nodes: %s", int((~keep).sum()),
                 out.loc[~keep, "common_name"].tolist())
        out = out[keep].reset_index(drop=True)
        nwk, _ = induced_subtree(out["ott_id"].tolist())
    tree = RefTree(nwk, [f"ott{i}" for i in out["ott_id"]])
    freeze(out, tree.to_newick(), "heldout_eco")
    log.info("held-out eco set: %d mammals (ForStrat: %s), %d birds", int((out.stratum == "mammals").sum()),
             out.loc[out.stratum == "mammals", "forstrat_mammal"].value_counts().to_dict(),
             int((out.stratum == "birds").sum()))
    return out
