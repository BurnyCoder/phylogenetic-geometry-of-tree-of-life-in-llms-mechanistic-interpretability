"""Reconstruct exploratory angular shortest paths, including zero-distance edges between names."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from phylogeo.config import RESULTS_DIR
from .statistics import MAIN, FrozenTree, distances, frame, read_json, save, log


def graph_distances(d):
    n = len(d)
    # Stable name-row order breaks ties at the k-th neighbour; exclude self by index.
    neighbours = [np.lexsort((np.arange(n), row)) for row in d]
    neighbours = np.stack([r[r != i] for i, r in enumerate(neighbours)])
    keep = np.zeros((n, n), bool)
    for k in range(1, n):
        j = neighbours[:, k-1]
        keep[np.arange(n), j] = True
        keep[j, np.arange(n)] = True
        a, b = np.nonzero(keep)
        graph = csr_matrix((d[a, b], (a, b)), shape=(n, n))
        if connected_components(graph, directed=False, return_labels=False) == 1:
            return k, shortest_path(graph, method="D", directed=False)
    raise ValueError("No connected graph on the supplied distances")


def run():
    df = frame()
    tree = FrozenTree([f"ott{k}" for k in df.ott_id])
    archive = read_json(RESULTS_DIR / MAIN / "stage2/results.json")
    i, j = np.triu_indices(len(df), 1)
    rows = []
    with threadpool_limits(limits=1):
        for layer, d in enumerate(distances(MAIN, df.common_name)):
            ang = np.arccos(np.clip(1 - d, -1, 1))
            k, geo = graph_distances(ang)
            rows.append({"layer": layer, "zero_pairs": int((ang[i, j] == 0).sum()), "k": k,
                         "archived_k": archive["geodesic_vs_angular"]["k"][layer],
                         "angular_rho": spearmanr(ang[i, j], tree.path[i, j]).statistic,
                         "geodesic_rho": spearmanr(geo[i, j], tree.path[i, j]).statistic,
                         "archived_geodesic_rho": archive["geodesic_vs_angular"]["geodesic_mantel"][layer]})
    leaves = frozenset(range(tree.n))
    splits = {min((clade, leaves - clade), key=lambda x: (len(x), sorted(x)))
              for members in tree.members.values() if 1 < len(members) < tree.n-1
              for clade in (frozenset(members),)}
    binary = tree.n-3
    rf = round(archive["nj_rf"]["model"] * 2 * binary)
    save("graph", {"by_layer": rows, "reference_splits": len(splits), "binary_nj_splits": binary,
                   "archive_symmetric_difference": rf, "implied_shared_splits": (binary + len(splits) - rf) // 2})
    log.info("Saved graph distances and resolved reference splits")
