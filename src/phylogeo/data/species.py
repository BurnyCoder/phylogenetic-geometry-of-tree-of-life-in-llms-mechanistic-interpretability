"""Reproducible selection of ~400 animal species with English common names.

Global context: implements the preregistered species rule (reports/preregistration.md
section 2.2). Candidates come from Wikidata (species rank, Animalia, English
Wikipedia article, NCBI taxid) through the QLever SPARQL endpoint (the public
Wikidata endpoint times out on this query). Candidates are matched to the Open
Tree Taxonomy, assigned to 13 coarse strata from their OTT lineage, ranked by
Wikipedia sitelinks within stratum, capped at 3 per family, filtered for
polysemous names, required to be leaves of the Open Tree synthetic tree, and
frozen (CSV + Newick + sha256) before any activation is extracted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from phylogeo.config import DATA_DIR, DATA_REPO_DIR
from phylogeo.webcache import request_text

log = logging.getLogger("phylogeo.species")

QLEVER = "https://qlever.dev/api/wikidata"  # fast SPARQL engine over Wikidata

# Strata in preregistered priority order: (stratum name, OTT lineage names that define it).
# Reptiles: OTT's "Reptilia" is flagged barren/inconsistent, so reptiles are the
# union of turtles, lepidosaurs (lizards, snakes, tuatara) and crocodilians.
STRATA: list[tuple[str, set[str]]] = [
    ("mammals", {"Mammalia"}),
    ("birds", {"Aves"}),
    ("reptiles", {"Testudines", "Lepidosauria", "Crocodylia"}),
    ("amphibians", {"Amphibia"}),
    ("ray-finned fish", {"Actinopterygii"}),
    ("cartilaginous fish", {"Chondrichthyes"}),
    ("insects", {"Insecta"}),
    ("arachnids", {"Arachnida"}),
    ("crustaceans", {"Crustacea"}),
    ("molluscs", {"Mollusca"}),
    ("cnidarians", {"Cnidaria"}),
    ("echinoderms", {"Echinodermata"}),
    ("other", {"Metazoa"}),
]

# Preregistered per-stratum quotas (sum = 400).
QUOTAS: dict[str, int] = {
    "mammals": 90, "birds": 70, "reptiles": 30, "amphibians": 20, "ray-finned fish": 45,
    "cartilaginous fish": 15, "insects": 45, "arachnids": 15, "crustaceans": 15,
    "molluscs": 20, "cnidarians": 10, "echinoderms": 10, "other": 15,
}

# Wikidata scientific names used to pre-assign candidates to strata (only to limit
# the number of Open Tree lookups; the final stratum always comes from OTT).
WIKIDATA_STRATUM_NAMES = {
    "Mammalia": "mammals", "Aves": "birds", "Testudines": "reptiles", "Squamata": "reptiles",
    "Crocodilia": "reptiles", "Rhynchocephalia": "reptiles", "Amphibia": "amphibians",
    "Actinopterygii": "ray-finned fish", "Chondrichthyes": "cartilaginous fish",
    "Insecta": "insects", "Arachnida": "arachnids", "Crustacea": "crustaceans",
    "Mollusca": "molluscs", "Cnidaria": "cnidarians", "Echinodermata": "echinoderms",
}

# Preregistered list of names with prominent non-animal senses (section 2.2, item 7).
POLYSEMY_LIST = {
    "jaguar", "cougar", "wolverine", "puma", "impala", "mustang", "falcon", "hornet", "viper",
    "cobra", "python", "lynx", "panther", "raven", "eagle", "swift", "kite", "crane", "seal",
    "bass", "sole", "turkey", "mole", "bat", "cardinal", "robin", "buffalo",
}

RANKS = ["phylum", "class", "order", "family", "genus"]

_Q_CANDIDATES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?item ?sci ?ncbi ?sl ?title ?label WHERE {
  ?item wdt:P105 wd:Q7432 .
  ?item wikibase:sitelinks ?sl .
  FILTER(?sl >= %(min_sl)d)
  ?item wdt:P171+ wd:Q729 .
  ?item wdt:P685 ?ncbi .
  ?item wdt:P225 ?sci .
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title .
  OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label) = "en") }
}
"""

_Q_STRATA = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?item ?sname WHERE {
  VALUES ?sname { %(names)s }
  ?s wdt:P225 ?sname .
  ?s wdt:P171* wd:Q729 .
  ?item wdt:P105 wd:Q7432 .
  ?item wikibase:sitelinks ?sl .
  FILTER(?sl >= %(min_sl)d)
  ?item wdt:P171+ ?s .
}
"""


def _sparql_csv(query: str) -> pd.DataFrame:
    """Run a SPARQL query on QLever and return the CSV result as a DataFrame."""
    text = request_text(QLEVER, namespace="qlever", params={"query": query})
    return pd.read_csv(io.StringIO(text), dtype=str)


def query_candidates(min_sitelinks: int = 15) -> pd.DataFrame:
    """All Wikidata animal species with >= ``min_sitelinks`` sitelinks, an NCBI id,
    a scientific name and an English Wikipedia article, plus a Wikidata stratum guess.

    Returns columns: wikidata, sci_name, ncbi, sitelinks, title, wd_stratum.
    """
    df = _sparql_csv(_Q_CANDIDATES % {"min_sl": min_sitelinks})
    df = df.rename(columns={"item": "wikidata", "sci": "sci_name", "sl": "sitelinks"})
    df["wikidata"] = df["wikidata"].str.replace("http://www.wikidata.org/entity/", "", regex=False)
    df["sitelinks"] = df["sitelinks"].astype(int)
    # Items with several NCBI ids / titles: keep the first row per item.
    df = df.drop_duplicates("wikidata").reset_index(drop=True)
    names = " ".join(f'"{n}"' for n in WIKIDATA_STRATUM_NAMES)
    mem = _sparql_csv(_Q_STRATA % {"names": names, "min_sl": min_sitelinks})
    mem["item"] = mem["item"].str.replace("http://www.wikidata.org/entity/", "", regex=False)
    order = [s for s, _ in STRATA]
    guess: dict[str, str] = {}
    for item, grp in mem.groupby("item"):
        strata = {WIKIDATA_STRATUM_NAMES[n] for n in grp["sname"]}
        guess[item] = min(strata, key=order.index)  # first in priority order
    df["wd_stratum"] = df["wikidata"].map(guess).fillna("other")
    log.info("Wikidata candidates: %d (by guessed stratum: %s)", len(df),
             dict(Counter(df["wd_stratum"])))
    return df


def _capitalized_words(titles: list[str]) -> set[str]:
    """Words that appear capitalized in a non-initial position of some title.

    Used to decide whether the title's first word is a proper noun (e.g.
    "African" in "African bush elephant") that must keep its capital letter.
    """
    caps: Counter = Counter()
    lows: Counter = Counter()
    for t in titles:
        words = t.split()
        for w in words[1:]:
            (caps if w[:1].isupper() else lows)[w.lower()] += 1
    return {w for w in caps if caps[w] > lows.get(w, 0)}


GEO_WORDS = {"north", "south", "east", "west", "central"}


def first_word_is_proper(first: str, single_word: bool, proper: set[str],
                         next_capitalized: bool = False) -> bool:
    """Decide whether the first word of a sentence-case Wikipedia title is a proper noun.

    Rules (in order): possessives ("Darwin's") are proper; if WordNet knows the
    word, it is proper iff all its matching lemmas are capitalized ("African" yes,
    "common" no); unknown words count as proper in multi-word names ("Komodo
    dragon", "Gila monster") unless they occur lowercase elsewhere, and as common
    nouns in single-word names ("heikegani").
    """
    from nltk.corpus import wordnet as wn  # lazy: needs the WordNet corpus

    base = first.split("-")[0].replace("'s", "").rstrip("'")
    low = base.lower()
    if low == "pacific" or (low in GEO_WORDS and next_capitalized):
        return True  # "Pacific oyster", "West Indian Ocean coelacanth", "South American lungfish"
    syns = wn.synsets(low)
    lemmas = [l.name() for syn in syns for l in syn.lemmas() if l.name().lower() == low]
    if lemmas:
        return all(l[:1].isupper() for l in lemmas)  # "African" yes; "common", "lion" no
    if syns:
        return False  # inflected common word ("basking" -> bask)
    if "'" in first or low in proper:
        return True  # possessive of an unknown name ("Hermann's"), or corpus-proper word
    return not single_word


def common_name(title: str, sci_name: str, proper: set[str]) -> str | None:
    """Turn an English Wikipedia title into a common name, or None if it is not one.

    Local: drops a trailing parenthetical ("Jaguar (animal)" -> "Jaguar"), rejects
    titles equal to the scientific name (the preregistered rule; a title equal to the
    genus, e.g. "Hippopotamus", IS the species' common name and is kept), keeps Wikipedia's casing
    for non-initial words (titles are sentence case, so only proper nouns are
    capitalized there) and lowercases the first word unless
    :func:`first_word_is_proper` says it is a proper noun.
    """
    name = re.sub(r"\s*\(.*\)\s*$", "", title).strip()
    if not name or any(ch.isdigit() for ch in name) or "," in name:
        return None
    if name.lower() == sci_name.lower():
        return None
    words = name.split()
    first = words[0]
    if not (len(first) > 1 and first[1:].isupper()) and not first_word_is_proper(first, len(words) == 1, proper,
                                                                    len(words) > 1 and words[1][:1].isupper()):
        name = first[0].lower() + name[1:]
    return name


def head_noun(name: str) -> str:
    """Last word of the common name, lowercased (used for lexical controls)."""
    return name.split()[-1].lower().replace("-", " ").split()[-1]


def polysemy_flag(name: str) -> bool:
    """True if a single-word name has a prominent non-animal sense (preregistered rule).

    Rule: name is in POLYSEMY_LIST, or it is a single word whose first WordNet
    noun sense is not in the ``noun.animal`` lexicographer file.
    """
    low = name.lower()
    if low in POLYSEMY_LIST:
        return True
    if " " in low or "-" in low:
        return False
    from nltk.corpus import wordnet as wn  # lazy: needs the WordNet corpus

    syns = wn.synsets(low, pos=wn.NOUN)
    return bool(syns) and syns[0].lexname() != "noun.animal"


def ensure_wordnet() -> None:
    """Download the NLTK WordNet corpus into DATA_DIR once (no-op if present)."""
    import nltk

    path = str(DATA_DIR / "nltk")
    if path not in nltk.data.path:
        nltk.data.path.insert(0, path)
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", download_dir=path, quiet=True)
        nltk.download("omw-1.4", download_dir=path, quiet=True)


def assign_stratum(lineage_names: set[str]) -> str | None:
    """First preregistered stratum whose defining taxa intersect the OTT lineage."""
    for name, taxa in STRATA:
        if lineage_names & taxa:
            return name
    return None  # not an animal (no Metazoa in lineage)


def lineage_ranks(lineage: list[tuple[str, str, int]]) -> dict[str, str]:
    """Pick phylum/class/order/family/genus names from an OTT lineage (nearest first)."""
    out = {r: "" for r in RANKS}
    for name, rank, _ in lineage:
        if rank in out and not out[rank]:
            out[rank] = name
    return out


def select_by_quota(df: pd.DataFrame, quotas: dict[str, int], family_cap: int = 3) -> pd.DataFrame:
    """Within each stratum take species by descending sitelinks, <= family_cap per family.

    Local: species with an unknown family use their genus (then their own name)
    as the family key, so they are not all lumped into one "family".
    Global: the core of the preregistered selection rule.
    """
    rows = []
    for stratum, q in quotas.items():
        cand = df[df["stratum"] == stratum].sort_values(["sitelinks", "sci_name"], ascending=[False, True])
        fam_count: Counter = Counter()
        taken = 0
        for _, r in cand.iterrows():
            fam = r["family"] or r["genus"] or r["sci_name"]
            if fam_count[fam] >= family_cap:
                continue
            fam_count[fam] += 1
            rows.append(r)
            taken += 1
            if taken >= q:
                break
        if taken < q:
            log.warning("stratum %s: only %d / %d species available", stratum, taken, q)
    return pd.DataFrame(rows).reset_index(drop=True)


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file (used to freeze inputs before analysis)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(df: pd.DataFrame, newick: str, name: str = "species") -> dict:
    """Write ``data/<name>.csv`` and ``data/<name>_tree.nwk`` and record their sha256."""
    csv_path = DATA_REPO_DIR / f"{name}.csv"
    nwk_path = DATA_REPO_DIR / f"{name}_tree.nwk"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    nwk_path.write_text(newick + "\n", encoding="utf-8", newline="\n")
    rec = {"csv": csv_path.name, "csv_sha256": sha256_file(csv_path),
           "tree": nwk_path.name, "tree_sha256": sha256_file(nwk_path), "n": int(len(df))}
    freeze_path = DATA_REPO_DIR / "FREEZE.json"
    allrec = json.loads(freeze_path.read_text()) if freeze_path.exists() else {}
    allrec[name] = rec
    freeze_path.write_text(json.dumps(allrec, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("froze %s: n=%d csv=%s tree=%s", name, len(df), rec["csv_sha256"][:12], rec["tree_sha256"][:12])
    return rec


# ---------------------------------------------------------------------------
# End-to-end builders (called by the "species" phase of phylogeo.run)
# ---------------------------------------------------------------------------

EXTINCT_FLAGS = {"extinct", "extinct_inherited"}


def annotate_with_ott(df: pd.DataFrame) -> pd.DataFrame:
    """Add OTT id, stratum, ranks and flags; drop unmatched, non-animal and extinct taxa.

    Local: TNRS on scientific names (context "Animals"), then one cached
    taxon_info call per matched id for the lineage.
    """
    from phylogeo.data.opentree import taxon_lineages, tnrs_match

    m = tnrs_match(df["sci_name"].tolist())
    df = df[df["sci_name"].isin(m)].copy()
    df["ott_id"] = df["sci_name"].map(lambda s: m[s]["ott_id"])
    lin = taxon_lineages(df["ott_id"].tolist())
    strata, ranks, extinct = [], [], []
    for oid in df["ott_id"]:
        rec = lin[oid]
        names = {n for n, _, _ in rec["lineage"]}
        strata.append(assign_stratum(names))
        ranks.append(lineage_ranks(rec["lineage"]))
        extinct.append(bool(EXTINCT_FLAGS & set(rec["flags"])))
    df["stratum"] = strata
    for r in RANKS:
        df[f"ott_{r}"] = [x[r] for x in ranks]
    df["extinct"] = extinct
    n0 = len(df)
    df = df[df["stratum"].notna() & ~df["extinct"]].drop_duplicates("ott_id")
    log.info("OTT-annotated: kept %d / %d (dropped non-animal/extinct/duplicate ott ids)", len(df), n0)
    # Linnaean ranks: NCBI Taxonomy first (complete for most animals), OTT as fallback
    # (OTT leaves e.g. Primates as "no rank"); see data/ncbi.py and preregistration deviations.
    from phylogeo.data.ncbi import lineage_ranks as ncbi_ranks

    nr = ncbi_ranks(df["ncbi"].astype(int).tolist())
    for r in RANKS:
        ncbi_vals = nr[r].to_numpy()
        df[r] = np.where(ncbi_vals != "", ncbi_vals, df[f"ott_{r}"].to_numpy())
        log.info("rank %s: %d from NCBI, %d OTT fallback, %d missing", r, int((ncbi_vals != "").sum()),
                 int(((ncbi_vals == "") & (df[f"ott_{r}"] != "")).sum()), int((df[r] == "").sum()))
    return df.drop(columns=["extinct"])


def build_species_set(min_sitelinks: int = 5, prefilter_factor: int = 6) -> pd.DataFrame:
    """Run the full preregistered selection and freeze ``data/species.csv`` + tree.

    Steps: Wikidata candidates -> common names -> prefilter by guessed stratum
    (top ``prefilter_factor`` x quota by sitelinks; ``other`` gets 4x more because
    Wikidata's lineage is incomplete) -> OTT annotation -> keep leaves of the
    synthetic tree -> polysemy flag -> quota selection with family cap -> freeze.
    Also freezes the polysemous sensitivity set (``data/species_polysemous.csv``).
    """
    from phylogeo.data.opentree import induced_subtree
    from phylogeo.data.tree import RefTree

    ensure_wordnet()
    cand = query_candidates(min_sitelinks)
    proper = _capitalized_words(cand["title"].tolist())
    cand["common_name"] = [common_name(t, s, proper) for t, s in zip(cand["title"], cand["sci_name"])]
    # Titles that needed a parenthetical qualifier (e.g. "Beluga (sturgeon)") have an
    # ambiguous bare name -> polysemy flag (deviation logged in the preregistration).
    cand["qualified_title"] = cand["title"].str.contains(r"\(", regex=True)
    cand = cand[cand["common_name"].notna()].copy()
    dup = cand["common_name"].str.lower().duplicated(keep=False)
    log.info("dropping %d rows with duplicated common names", int(dup.sum()))
    cand = cand[~dup]
    # Prefilter (only to limit API calls; final strata come from OTT).
    parts = []
    for stratum, q in QUOTAS.items():
        k = q * prefilter_factor * (4 if stratum == "other" else 1)
        parts.append(cand[cand["wd_stratum"] == stratum].nlargest(k, "sitelinks"))
    pre = pd.concat(parts).drop_duplicates("wikidata")
    log.info("prefiltered candidates: %d", len(pre))
    ann = annotate_with_ott(pre)
    # Keep only taxa usable as leaves of the synthetic tree.
    _, dropped = induced_subtree(ann["ott_id"].tolist())
    ann = ann[~ann["ott_id"].isin(dropped)].copy()
    log.info("in synthetic tree: %d (dropped %d)", len(ann), len(dropped))
    ann["polysemous"] = [polysemy_flag(n) or bool(q) for n, q in zip(ann["common_name"], ann["qualified_title"])]
    ann["n_words"] = ann["common_name"].str.split().str.len()
    ann["head_noun"] = ann["common_name"].map(head_noun)
    primary = select_by_quota(ann[~ann["polysemous"]], QUOTAS)
    with_poly = select_by_quota(ann, QUOTAS)
    poly = with_poly[with_poly["polysemous"]].reset_index(drop=True)
    cols = ["common_name", "sci_name", "wikidata", "ncbi", "ott_id", "sitelinks", "stratum",
            *RANKS, *[f"ott_{r}" for r in RANKS], "n_words", "head_noun", "polysemous"]
    out = {}
    for name, d in (("species", primary), ("species_polysemous", poly)):
        d = d[cols].reset_index(drop=True)
        labels = [f"ott{i}" for i in d["ott_id"]]
        newick, dropped2 = induced_subtree(d["ott_id"].tolist())
        assert not dropped2, f"unexpected dropped ids in final tree: {dropped2}"
        tree = RefTree(newick, labels)  # validates leaves == species, suppresses unifurcations
        freeze(d, tree.to_newick(), name)
        out[name] = d
    log.info("primary set by stratum: %s", dict(Counter(out["species"]["stratum"])))
    log.info("single-word names: %.1f%%", 100 * (out["species"]["n_words"] == 1).mean())
    return out["species"]


def build_conflict_set() -> pd.DataFrame:
    """Resolve the preregistered conflict-triplet names to OTT and freeze their tree.

    Writes ``data/conflict.csv`` (one row per unique name) and ``data/conflict_tree.nwk``.
    Names whose scientific name fails to match are kept with ott_id = -1 and any
    triplet that uses them is later excluded as unresolved.
    """
    from phylogeo.data.opentree import induced_subtree, tnrs_match
    from phylogeo.data.tree import RefTree

    trip = pd.read_csv(DATA_REPO_DIR / "conflict_triplets.csv")
    names: dict[str, str] = {}
    for role in ("anchor", "phylo", "folk"):
        for n, s in zip(trip[role], trip[f"{role}_sci"]):
            names[n] = s
    df = pd.DataFrame({"common_name": list(names), "sci_name": list(names.values())})
    m = tnrs_match(df["sci_name"].tolist())
    df["ott_id"] = [m[s]["ott_id"] if s in m else -1 for s in df["sci_name"]]
    # Names that resolve below species rank (e.g. dog -> Canis lupus familiaris) are
    # placed at their species (the nearest species-rank ancestor in the OTT lineage),
    # so every leaf of the conflict tree is a species.
    from phylogeo.data.opentree import taxon_lineage

    for k, (s, oid) in enumerate(zip(df["sci_name"], df["ott_id"])):
        if oid > 0 and m[s]["rank"] not in ("species", None):
            sp = [o for _, r, o in taxon_lineage(oid)["lineage"] if r == "species"]
            if sp:
                log.info("conflict set: %s (%s) placed at species ott%d", s, m[s]["rank"], sp[0])
                df.loc[k, "ott_id"] = sp[0]
    ids = sorted({i for i in df["ott_id"] if i > 0})
    newick, dropped = induced_subtree(ids)
    if dropped:
        log.info("conflict set: not usable as leaves: %s", df[df["ott_id"].isin(dropped)]["common_name"].tolist())
    df.loc[df["ott_id"].isin(dropped), "ott_id"] = -1
    leaves = sorted({i for i in df["ott_id"] if i > 0})
    tree = RefTree(newick, [f"ott{i}" for i in leaves])
    freeze(df, tree.to_newick(), "conflict")
    log.info("conflict set: %d names, %d unresolved", len(df), int((df["ott_id"] < 0).sum()))
    return df
