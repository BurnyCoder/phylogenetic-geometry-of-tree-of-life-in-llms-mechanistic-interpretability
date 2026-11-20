"""Single entry point: ``python -m phylogeo.run --config configs/X.yaml --phases a,b,c``.

Global context: wraps the pipeline phases, each implemented in its own module:
  species  -> data/species.py   build + freeze species set, polysemous set, conflict set
  dates    -> data/tree.py      date the reference tree with TimeTree (local cache only)
  extract  -> extract.py        residual-stream activations for a model (per config)
  pca      -> phases.py         stage 1a: top-3 PCA plots + H1a checks
  power    -> phases.py         power simulation on the frozen tree
  analyze  -> phases.py         stage 1b: triplet accuracy, Mantel, baselines, controls
  report   -> report/           regenerate all tables/figures from results/
Every phase logs to logs/<timestamp>_<run>.log and writes to results/<run>/.
"""

from __future__ import annotations

import argparse
import logging
import sys

from phylogeo.config import RunConfig, load_config, setup_logging

log = logging.getLogger("phylogeo.run")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=False, help="YAML run config (configs/*.yaml)")
    ap.add_argument("--phases", required=True, help="comma-separated phases to run in order")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else RunConfig(name="data")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow any token text
    setup_logging(cfg.name)
    log.info("config: %s", cfg)
    for phase in args.phases.split(","):
        log.info("=== phase %s ===", phase)
        if phase == "species":
            from phylogeo.data.species import build_conflict_set, build_species_set

            build_species_set()
            build_conflict_set()
        else:
            from phylogeo import phases

            fn = getattr(phases, f"phase_{phase}", None)
            if fn is None:
                raise SystemExit(f"unknown phase: {phase}")
            fn(cfg)
        log.info("=== phase %s done ===", phase)


if __name__ == "__main__":
    main()
