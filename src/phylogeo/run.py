"""Single entry point: ``python -m phylogeo.run --config configs/X.yaml --phases a,b,c``.

Global context: wraps the pipeline phases, each implemented in its own module:
  species  -> data/species.py   build + freeze species set, polysemous set, conflict set
  dates    -> data/tree.py      date the reference tree with TimeTree (local cache only)
  extract  -> extract.py        residual-stream activations for a model (per config)
  pca      -> phases.py         stage 1a: top-3 PCA plots + H1a checks
  power    -> phases.py         power simulation on the frozen tree
  analyze  -> analysis.py       stage 1b: triplet accuracy, Mantel, baselines, controls
  conflict -> analysis.py       phylogeny-vs-folk conflict triplets
  stage2   -> stage2.py         geodesic vs angular, delta-hyperbolicity, NJ tree, flat probe
  stage3   -> stage3.py         behavior + causal ablation/interchange (stage3_causal: causal only)
  stage4   -> stage4.py         ablations (readout, normalization, format, names, probe variants)
  eco      -> eco.py            Addendum A: habitat axis on held-out species
  scale    -> stage5.py         stage 5 per-run summary (scale_summary: all runs + figures)
  stage6   -> stage6.py         sensitivity to seeds, templates, species subsamples
  audit    -> audit.py          independent re-derivation and stress tests of headline numbers
  report   -> report.py         regenerate all tables, RESULTS.md and README/summary blocks
  site     -> site.py           static GitHub Pages explorer in site/
  paper    -> paper.py          reports/paper.typ -> reports/paper.pdf (Typst)
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
    ap.add_argument("--name-sets", default=None, help="override name_sets for the extract phase")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else RunConfig(name="data")
    if args.name_sets:
        cfg.extra["name_sets"] = args.name_sets.split(",")
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
