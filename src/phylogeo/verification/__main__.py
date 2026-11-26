"""Reproduce the scientific audit from frozen observations or publish from checked-in statistics."""

import argparse

from phylogeo.config import RESULTS_DIR, setup_logging
from .statistics import MAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="use the committed statistical replay files")
    parser.add_argument("--reuse-unchanged-media", action="store_true", help="reuse existing unaffected static illustrations")
    args = parser.parse_args()
    setup_logging("scientific_verification")
    from .sources import original_observations, write_index
    original_observations()
    if not args.render_only:
        from . import primary, dating, outcomes, scale, graph
        for analysis in (primary, dating, outcomes, scale, graph):
            analysis.run()

    from phylogeo.report import write_results_md
    from phylogeo.site import build_site
    from phylogeo.paper import build_paper
    from .cards import write_cards
    runs = sorted(p.name for p in RESULTS_DIR.iterdir() if (p / "stage1a").exists() or (p / "stage1b").exists())
    write_results_md(runs, MAIN)
    build_site(MAIN, reuse_media=args.reuse_unchanged_media)
    build_paper()
    write_cards()
    write_index()


if __name__ == "__main__":
    main()
