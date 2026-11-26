"""Replay each published scale/checkpoint row from its own archived species activations."""

import hashlib

import numpy as np
from threadpoolctl import threadpool_limits

from phylogeo.config import RESULTS_DIR
from .statistics import FrozenTree, crossfit, distances, frame, log, node_weights, read_json, save, scores, stream, triplet_test


def run():
    df = frame()
    tree = FrozenTree([f"ott{k}" for k in df.ott_id])
    t = tree.triplets()
    a, b, c = t[:3]
    h, g = df.head_noun.to_numpy(), df.stratum.to_numpy()
    t = t[:, (g[a] == g[b]) & (g[b] == g[c]) & (g[a] != "other") &
          (h[a] != h[b]) & (h[a] != h[c]) & (h[b] != h[c])]
    fam = np.where(df.family != "", df.family, df.genus)
    rows = {}
    with threadpool_limits(limits=1):
        for path in sorted(RESULTS_DIR.glob("*/stage5/scale.json")):
            old = read_json(path)
            run = old["run"]
            d = distances(run, df.common_name)
            test = triplet_test(d, t, g, "maxT", n=199)
            cf, L = crossfit(scores(d, t), t, fam, stream("cf"), 10)
            historical = scores(d, t, True) @ node_weights(t[3])
            rows[run] = {"revision": old["revision"], "step": old["step"], "nominal_params": old["params"],
                         "crossfit": cf, "layer": L, "p": test["p"], "by_layer": test["by_layer"],
                         "matrix_sha256": hashlib.sha256(d.tobytes()).hexdigest(),
                         "historical_max_abs_error": np.max(abs(historical - old["ta_within_by_layer"]))}
            log.info("Replayed scale row %s", run)
    save("scale", rows)


if __name__ == "__main__":
    from phylogeo.config import setup_logging

    setup_logging("verify_scale")
    run()
