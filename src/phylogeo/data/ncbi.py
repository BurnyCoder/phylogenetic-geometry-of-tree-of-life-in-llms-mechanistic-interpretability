"""Linnaean ranks from the NCBI Taxonomy dump (fills gaps in Open Tree Taxonomy ranks).

Global context: the Open Tree Taxonomy leaves many ranks unset (e.g. Primates is
"no rank"; coelacanth ranks are all unset), which mis-bins pairs in rank-based
analyses (S2, S5, rank oracle, depth profile, family blocks). NCBI Taxonomy
(public domain; https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/, format described in
taxdump_readme.txt) assigns phylum/class/order/family/genus to most animals.
Ranks are taken from NCBI first and from OTT where NCBI has none.
"""

from __future__ import annotations

import logging
import tarfile

import pandas as pd  # noqa: F401  (DataFrame output)
import requests

from phylogeo.config import DATA_DIR

log = logging.getLogger("phylogeo.ncbi")
URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
RANKS = ["phylum", "class", "order", "family", "genus"]


def _dump_dir():
    d = DATA_DIR / "ncbi_taxdump"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "nodes.dmp").exists():
        tgz = d / "taxdump.tar.gz"
        log.info("downloading %s", URL)
        with requests.get(URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tgz, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with tarfile.open(tgz) as t:
            for name in ("nodes.dmp", "names.dmp", "merged.dmp"):
                t.extract(name, d, filter="data")
    if not (d / "merged.dmp").exists():  # older cache without merged.dmp
        with tarfile.open(d / "taxdump.tar.gz") as t:
            t.extract("merged.dmp", d, filter="data")
    return d


def lineage_ranks(taxids: list[int]) -> pd.DataFrame:
    """phylum/class/order/family/genus names (NCBI scientific names) for each taxid."""
    d = _dump_dir()
    # nodes.dmp / names.dmp fields are separated by the literal string "\t|\t" and
    # lines end with "\t|" (taxdump_readme.txt); parse with plain string splitting.
    parent, rank = {}, {}
    with open(d / "nodes.dmp", encoding="utf-8") as f:
        for line in f:
            p = line.split("\t|\t", 3)
            parent[int(p[0])] = int(p[1])
            rank[int(p[0])] = p[2]
    # merged.dmp: old_tax_id | new_tax_id  (Wikidata may hold ids NCBI has since merged)
    merged = {}
    with open(d / "merged.dmp", encoding="utf-8") as f:
        for line in f:
            p = line.split("	|	")
            merged[int(p[0])] = int(p[1].split("	")[0])
    sci = {}
    with open(d / "names.dmp", encoding="utf-8") as f:
        for line in f:
            if "scientific name" in line:
                p = line.split("\t|\t")
                sci[int(p[0])] = p[1]
    rows = []
    for t in taxids:
        rec = {r: "" for r in RANKS}
        cur, seen = merged.get(int(t), int(t)), 0
        while cur in parent and cur != 1 and seen < 100:
            if rank.get(cur) in rec and not rec[rank[cur]]:
                rec[rank[cur]] = sci.get(cur, "")
            cur = parent[cur]
            seen += 1
        rows.append(rec)
    return pd.DataFrame(rows)
