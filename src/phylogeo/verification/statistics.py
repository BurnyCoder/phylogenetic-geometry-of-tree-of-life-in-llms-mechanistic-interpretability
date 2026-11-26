"""Scoring, resampling, and provenance for an independent reanalysis of archived data."""

import hashlib
import itertools
import json
import logging
import zlib
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd

from phylogeo.config import ACT_DIR, DATA_REPO_DIR, RESULTS_DIR

MAIN = "qwen35_2b"
CONTROLS = ("qwen35_2b_random", "pythia410m_step0")
OUT = RESULTS_DIR / "scientific-verification"
log = logging.getLogger("phylogeo.verification")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(name, value):
    """Atomic replacement avoids presenting an interrupted calculation as complete."""
    OUT.mkdir(exist_ok=True)
    dest = OUT / (name + ".json")
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.tolist()) + "\n",
                    encoding="utf-8")
    temp.replace(dest)


def stream(tag):
    """Recover the independent seed streams documented in the original run."""
    return np.random.default_rng([0, zlib.crc32(tag.encode())])


def pair_scores(sister, first, second, historical=False):
    """Score the correct pair against both alternatives, with an explicit minimum-tie convention."""
    low = np.minimum(np.minimum(sister, first), second)
    count = (sister == low).astype(int) + (first == low) + (second == low)
    if historical:
        return (sister == low) / count
    return np.where(count > 1, 1 / 3, (sister == low).astype(float))


def transition_bounds(n, before, after):
    """Sharp bounds for the size of after minus before, given two marginal counts."""
    if not 0 <= before <= n or not 0 <= after <= n:
        raise ValueError("Counts must be between zero and the number of paired trials")
    return [max(0, after - before), min(after, n - before)]


def holm_family(values):
    """Holm step-down with the full registered family size, including unavailable slots."""
    out = dict.fromkeys(values)
    last = 0.
    for i, (k, p) in enumerate(sorted(((k, p) for k, p in values.items() if p is not None),
                                     key=lambda kv: kv[1])):
        last = min(1., max(last, (len(values) - i) * p))
        out[k] = last
    return out


def frame(name="species"):
    return pd.read_csv(DATA_REPO_DIR / f"{name}.csv", keep_default_na=False)


def check_freeze():
    frozen = read_json(DATA_REPO_DIR / "FREEZE.json")
    out = {}
    for k, entry in frozen.items():
        ids = frame(k)
        if len(ids) != entry["n"]:
            raise ValueError(f"{k}: frozen row count differs")
        out[k] = {"n_rows": len(ids), "unique_ott": ids.ott_id.nunique()}
        for field in ("csv", "tree"):
            digest = hashlib.sha256((DATA_REPO_DIR / entry[field]).read_bytes()).hexdigest()
            if digest != entry[field + "_sha256"]:
                raise ValueError(f"Freeze mismatch for {entry[field]}")
            out[k][field + "_sha256"] = digest
    main = frame()
    for k in ("conflict", "heldout_eco", "species_polysemous"):
        subset = frame(k)
        out[k]["ott_overlap_primary"] = len(set(subset.ott_id) & set(main.ott_id))
        for field in ("family", "order"):
            if field not in subset:
                continue
            out[k][field + "_overlap_primary"] = len((set(subset[field]) & set(main[field])) - {""})
    return out


class FrozenTree:
    """Assign each leaf pair to its deepest enclosing clade by overwriting in root-first order."""

    def __init__(self, names, name="species"):
        tree = dendropy.Tree.get(path=str(DATA_REPO_DIR / f"{name}_tree.nwk"), schema="newick",
                                 rooting="force-rooted")
        leaves = {n.taxon.label: n for n in tree.leaf_node_iter()}
        if set(leaves) != set(names) or len(names) != len(leaves):
            raise ValueError("The frozen tree and ordered unique OTT identifiers differ")
        index = {leaves[label]: i for i, label in enumerate(names)}
        self.n = len(names)
        self.children = {}
        self.members = {}
        for n in tree.postorder_node_iter():
            if n not in index:
                index[n] = len(index)
                self.children[index[n]] = [index[c] for c in n.child_nodes()]
            self.members[index[n]] = np.array([index[leaf] for leaf in n.leaf_iter()])
        self.depth = np.empty(len(index), dtype=int)
        self.lca = np.empty((self.n, self.n), dtype=int)
        for n in tree.preorder_node_iter():
            i = index[n]
            self.depth[i] = 0 if n.parent_node is None else self.depth[index[n.parent_node]] + 1
            leaves_i = self.members[i]
            self.lca[np.ix_(leaves_i, leaves_i)] = i
        tips = self.depth[:self.n]
        self.path = tips[:, None] + tips - 2 * self.depth[self.lca]

    def triplets(self, tag="triplets", cap=200):
        """Uniform with replacement for large nodes, exhaustive for small nodes, in frozen order."""
        rng = stream(tag)
        rows = []
        for node, kids in self.children.items():
            groups = [self.members[k] for k in kids]
            sizes = np.array([len(g) for g in groups])
            counts = sizes * (sizes - 1) // 2 * (sum(sizes) - sizes)
            if not counts.sum():
                continue
            for i in (range(len(groups)) if counts.sum() <= cap else
                      rng.choice(len(groups), cap, p=counts / counts.sum())):
                if not counts[i]:
                    continue
                outside = np.concatenate([g for j, g in enumerate(groups) if j != i])
                if counts.sum() <= cap:
                    rows.extend((a, b, c, node) for a, b in itertools.combinations(groups[i], 2) for c in outside)
                else:
                    a, b = rng.choice(groups[i], 2, replace=False)
                    rows.append((a, b, rng.choice(outside), node))
        trip = np.array(rows, dtype=int).T
        a, b, c, n = trip
        if not (np.all(self.lca[a, c] == n) and np.all(self.lca[b, c] == n)
                and np.all(self.depth[self.lca[a, b]] > self.depth[n])):
            raise ValueError("A sampled triple violates the rooted MRCA definition")
        return trip


def distances(run, names, name_set="species"):
    """Read the archive sequentially for spinning disks. Reproduce float32 centering and cosine per layer."""
    directory = ACT_DIR / run / name_set
    if (directory / "names.txt").read_text(encoding="utf-8").splitlines() != list(names):
        raise ValueError(f"Misaligned name rows in {directory}")
    raw = np.load(directory / "last.npy", allow_pickle=False)
    if raw.ndim != 4 or raw.shape[:2] != (len(names), 8) or raw.dtype != np.float32:
        raise ValueError(f"Unexpected archive shape or precision in {directory}")
    out = []
    for layer in range(raw.shape[2]):
        x = np.ascontiguousarray(raw[:, :, layer, :])
        v = (x - x.mean(axis=0)).mean(axis=1)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        d = np.clip(1 - v @ v.T, 0, 2)
        if not np.isfinite(d).all():
            raise ValueError(f"Nonfinite distances in {run}, layer {layer}")
        np.fill_diagonal(d, 0)
        out.append(d)
    return np.stack(out)


def permutations(groups, n, rng):
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for _ in range(n):
        p = np.arange(len(groups))
        for block in blocks:
            p[block] = rng.permutation(block)
        yield p


def scores(d, t, historical=False):
    a, b, c = t[:3]
    return pair_scores(d[..., a, b], d[..., a, c], d[..., b, c], historical)


def node_weights(nodes, weights=None, multiplicity=None):
    """Return one normalized contribution per triplet, after averaging within each decision node."""
    w = np.ones(len(nodes)) if weights is None else weights
    totals = np.bincount(nodes, weights=w)
    active = np.flatnonzero(totals)
    mult = np.ones(len(totals)) if multiplicity is None else multiplicity[:len(totals)]
    return w / np.maximum(totals[nodes], 1) * mult[nodes] / mult[active].sum()


def crossfit(s, t, families, rng, n_splits=20):
    keys = np.unique(families)
    values, selected = [], []
    a, b, c, node = t
    for _ in range(n_splits):
        train = np.isin(families, rng.permutation(keys)[:len(keys)//2])
        masks = [g[a] & g[b] & g[c] for g in (train, ~train)]
        if all(m.any() for m in masks):
            means = [s[:, m] @ node_weights(node[m]) for m in masks]
            for fit, test in (means, means[::-1]):
                best = int(fit.argmax())
                selected.append(best)
                values.append(test[best])
    return float(np.mean(values)), int(np.bincount(selected).argmax())


def tails(observed, null):
    return (1 + np.count_nonzero(np.asarray(null) >= observed)) / (1 + len(null))


def triplet_test(d, t, groups, tag, *, historical=False, heads=None, n=1000):
    """Fixed-mask max-T, or transport the head labels with vectors on prefiltered stratum triples."""
    a, b, c, node = t
    def basis(p):
        if heads is None:
            return fixed
        h = heads[p]
        keep = (h[a] != h[b]) & (h[a] != h[c]) & (h[b] != h[c])
        return node_weights(node, keep.astype(float))
    fixed = node_weights(node)
    ident = np.arange(len(groups))
    profile = scores(d, t, historical) @ basis(ident)
    maxima = []
    for p in permutations(groups, n, stream(tag)):
        ss = pair_scores(d[:, p[a], p[b]], d[:, p[a], p[c]], d[:, p[b], p[c]], historical)
        maxima.append((ss @ basis(p)).max())
    return {"by_layer": profile, "p": tails(profile.max(), maxima), "permutations": n}


def bootstrap(s, t, labels, tree, tag, n=1000, distinct_blocks=False):
    """Paired family/order resampling. Every replicate uses the same draw for all comparisons."""
    _, ids = np.unique(labels, return_inverse=True)
    n_blocks = ids.max() + 1
    block_of_node = np.full(len(tree.depth), -1)
    for v in np.unique(t[3]):
        affected = np.unique(ids[tree.members[v]])
        if len(affected) == 1:
            block_of_node[v] = affected[0]
    a, b, c, nodes = t
    estimates = []
    rng = stream(tag)
    for _ in range(n):
        counts = np.bincount(rng.integers(n_blocks, size=n_blocks), minlength=n_blocks)
        w = counts[ids]
        mult = np.where(block_of_node >= 0, counts[block_of_node], 1.)
        weights = w[a] * w[b] * w[c]
        if distinct_blocks:
            # Count a resampled family once if several tips in this triple belong to it.
            weights = w[a] * np.where(ids[b] == ids[a], 1, w[b]) * np.where(
                (ids[c] == ids[a]) | (ids[c] == ids[b]), 1, w[c])
        if weights.any():
            estimates.append(s @ node_weights(nodes, weights, mult))
    return np.stack(estimates).T
