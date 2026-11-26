"""Layer-adjusted rank/time test on saved observations. No tree ages leave the local cache."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from phylogeo.config import DATA_DIR
from .statistics import MAIN, FrozenTree, distances, frame, log, permutations, save, stream, tails


def unit(x):
    centered = x - x.mean()
    norm = np.linalg.norm(centered)
    if not norm:
        raise ValueError("A constant rank variable has undefined correlation")
    return centered / norm


def centered_ranks(keys, width):
    """Midranks from integer bins. Tied distances share the mean of their occupied ranks."""
    if len(keys) < 2000:
        return rankdata(keys) - (len(keys) + 1) / 2
    counts = np.bincount(keys, minlength=width)
    return (np.cumsum(counts) - .5 * counts - .5 * len(keys))[keys]


def evaluate_time(d, tree, df, age, draws=1000, rng=None):
    """Same permutation at every layer. Compare max |pooled partial rho| to its own null."""
    ii, jj = np.triu_indices(len(df), 1)
    ages = age[tree.lca[ii, jj]]
    depths = tree.depth[tree.lca[ii, jj]]
    level = np.full(len(ii), -1)
    for k, label in enumerate(["phylum", "class", "order", "family", "genus"]):
        x = df[label].to_numpy()
        level[(x[ii] != "") & (x[ii] == x[jj])] = k
    bins = []
    for k in np.unique(level):
        m = np.isfinite(ages) & (level == k)
        if m.sum() < 30 or np.unique(ages[m]).size < 2:
            continue
        age_unit = unit(rankdata(ages[m]))
        dep = rankdata(depths[m])
        dep_unit = unit(dep) if np.ptp(dep) else np.zeros(len(dep))
        residual = age_unit - np.dot(age_unit, dep_unit) * dep_unit
        if np.linalg.norm(residual) < 1e-10:
            raise ValueError("Age and MRCA depth are collinear in a contributing rank bin")
        bins.append((ii[m], jj[m], age_unit, dep_unit, residual / np.linalg.norm(residual)))
    perms = [np.arange(len(df)), *permutations(df.stratum.to_numpy(), draws, stream("S5") if rng is None else rng)]
    weights = np.array([len(row[0]) for row in bins], dtype=float)
    weights /= weights.sum()

    def layer_calc(layer):
        # Integer compression preserves all numerical ties; a histogram replaces expensive repeated sorts.
        vals, keys = np.unique(d[layer], return_inverse=True)
        keys = keys.reshape(d.shape[1:]).astype(np.int32)
        raw = np.zeros(len(perms))
        partial = np.zeros(len(perms))
        for i, p in enumerate(perms):
            for weight, (x, y, age_u, dep_u, residual_u) in zip(weights, bins):
                r = centered_ranks(keys[p[x], p[y]], len(vals))
                norm2 = r @ r
                projection = r @ dep_u
                raw[i] += weight * (r @ age_u) / np.sqrt(norm2)
                partial[i] += weight * (r @ residual_u) / np.sqrt(norm2 - projection**2)
        log.info("Time test completed layer %s", layer)
        return raw, partial

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as workers:
        all_stats = list(workers.map(layer_calc, range(len(d))))
    out = {"n_perm": draws, "dated_pairs_in_bins": sum(len(b[0]) for b in bins),
           "missing_pairs": int((~np.isfinite(ages)).sum()), "bin_sizes": [len(b[0]) for b in bins]}
    for j, key in enumerate(("raw", "partial")):
        values = np.stack([r[j] for r in all_stats])
        obs = values[:, 0]
        nullmax = np.abs(values[:, 1:]).max(axis=0)
        out[key] = {"by_layer": obs, "best_absolute_layer": int(np.abs(obs).argmax()),
                    "p_global_max": tails(np.abs(obs).max(), nullmax),
                    "p_fixed_by_layer": [tails(abs(v), abs(r[1:])) for v, r in zip(obs, values)],
                    "p_adjusted_by_layer": [tails(abs(v), nullmax) for v in obs]}
    return out


def run():
    df = frame()
    tree = FrozenTree([f"ott{k}" for k in df.ott_id])
    path = DATA_DIR / "timetree" / "species_node_ages.npy"
    age = np.load(path, allow_pickle=False)
    if len(age) != len(tree.depth):
        raise ValueError("TimeTree age indexing does not match the frozen tree")
    result = evaluate_time(distances(MAIN, df.common_name), tree, df, age)
    result["node_age_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save("dating", result)


if __name__ == "__main__":
    from phylogeo.config import setup_logging

    setup_logging("verify_dating")
    run()
