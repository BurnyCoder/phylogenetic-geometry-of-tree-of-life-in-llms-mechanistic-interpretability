"""Open Tree of Life API v3 client: name matching, taxonomy lineages, induced subtrees.

Global context: the reference phylogeny for every analysis is the Open Tree of Life
synthetic tree (CC0, https://tree.opentreeoflife.org/about/licenses). Species
names (scientific, from Wikidata) are matched to Open Tree Taxonomy (OTT) ids,
their Linnaean lineage gives strata/ranks/families, and the induced subtree over
the selected ids is the topology used for triplet accuracy and tree distances.
API docs: https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3
and https://github.com/OpenTreeOfLife/germinator/wiki/Taxonomy-API-v3
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from phylogeo.webcache import request_json

log = logging.getLogger("phylogeo.opentree")
API = "https://api.opentreeoflife.org/v3"


def tnrs_match(names: list[str], context: str = "Animals", chunk: int = 250) -> dict[str, dict]:
    """Match scientific names to OTT taxa with the TNRS service.

    Local: POST /tnrs/match_names in chunks; keeps only exact, non-approximate,
    non-synonym-ambiguous matches (first match with ``is_approximate_match`` False).
    Returns {name: {"ott_id", "unique_name", "rank", "flags"}}; unmatched names are absent.
    Global: first step of mapping Wikidata species onto the Open Tree.
    """
    out: dict[str, dict] = {}
    for i in range(0, len(names), chunk):
        batch = names[i : i + chunk]
        data = request_json(
            "POST",
            f"{API}/tnrs/match_names",
            namespace="ott_tnrs",
            payload={"names": batch, "context_name": context, "do_approximate_matching": False},
        )
        for res in data.get("results", []):
            exact = [m for m in res.get("matches", []) if not m.get("is_approximate_match")]
            if not exact:
                continue
            t = exact[0]["taxon"]
            out[res["name"]] = {
                "ott_id": int(t["ott_id"]),
                "unique_name": t.get("unique_name", t.get("name")),
                "rank": t.get("rank"),
                "flags": t.get("flags", []),
            }
    log.info("TNRS matched %d / %d names", len(out), len(names))
    return out


def taxon_lineage(ott_id: int) -> dict:
    """Return taxon info with lineage for one OTT id (cached).

    Local: POST /taxonomy/taxon_info with include_lineage=True.
    Returns {"ott_id", "name", "rank", "flags", "lineage": [(name, rank, ott_id), ...]}
    where lineage runs from the parent up to the root ("life").
    """
    d = request_json(
        "POST",
        f"{API}/taxonomy/taxon_info",
        namespace="ott_taxon_info",
        payload={"ott_id": int(ott_id), "include_lineage": True},
        accept_error_json=True,
    )
    if "lineage" not in d:  # unknown/deprecated id -> empty record
        return {"ott_id": int(ott_id), "name": None, "rank": None, "flags": [], "lineage": []}
    return {
        "ott_id": int(ott_id),
        "name": d.get("name"),
        "rank": d.get("rank"),
        "flags": d.get("flags", []),
        "lineage": [(x["name"], x["rank"], int(x["ott_id"])) for x in d["lineage"]],
    }


def taxon_lineages(ott_ids: list[int], workers: int = 8) -> dict[int, dict]:
    """Fetch lineages for many ids in parallel threads (each call is cached)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(taxon_lineage, ott_ids))
    return {r["ott_id"]: r for r in recs}


def induced_subtree(ott_ids: list[int], max_rounds: int = 10) -> tuple[str, list[int]]:
    """Get the synthetic-tree topology induced by ``ott_ids`` as Newick.

    Local: POST /tree_of_life/induced_subtree with id labels. Ids that are not in
    the synthetic tree come back in an "unknown" map (HTTP 400); they are removed
    and the call retried. "Broken" taxa (mapped to an MRCA node instead of a leaf)
    are also removed so that every leaf is a requested species.
    Returns (newick, dropped_ids).
    Global: produces the reference topology for all tree-based metrics.
    """
    ids = sorted({int(i) for i in ott_ids})
    dropped: list[int] = []
    for _ in range(max_rounds):
        d = request_json(
            "POST",
            f"{API}/tree_of_life/induced_subtree",
            namespace="ott_induced",
            payload={"ott_ids": ids, "label_format": "id"},
            accept_error_json=True,
        )
        bad: set[int] = set()
        for key in d.get("unknown", {}) or {}:
            bad.add(int(str(key).replace("ott", "")))
        for key in d.get("broken", {}) or {}:
            bad.add(int(str(key).replace("ott", "")))
        if "newick" in d and not bad:
            return d["newick"], dropped
        if not bad:
            raise RuntimeError(f"induced_subtree failed: {str(d)[:300]}")
        log.info("induced_subtree: dropping %d ids not usable as leaves", len(bad))
        dropped += sorted(bad)
        ids = [i for i in ids if i not in bad]
    raise RuntimeError("induced_subtree: too many rounds of dropping ids")
