"""Paths, run configuration and logging.

Global context: every phase of the pipeline (species -> tree -> extract -> pca ->
analyze -> report) reads its settings from a YAML file in ``configs/`` through
:func:`load_config`, writes small results to ``results/`` and large arrays to
``DATA_DIR`` (set in ``.env``; on this machine D:/AI/phylo-geometry because C: is
nearly full). Logging goes to a timestamped file in ``logs/`` and to the terminal.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repository root = two levels above this file (src/phylogeo/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
# Load .env once at import so HF_HOME etc. are set before transformers is imported.
load_dotenv(REPO_ROOT / ".env")

# Large-artifact directory (activations, API caches, non-redistributable TimeTree data).
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "cache"))
CACHE_DIR = DATA_DIR / "cache"  # JSON caches of web API responses
ACT_DIR = DATA_DIR / "activations"  # .npy activation arrays per run
RESULTS_DIR = REPO_ROOT / "results"  # small, git-tracked results (json/csv/png/html)
LOGS_DIR = REPO_ROOT / "logs"  # timestamped log files
DATA_REPO_DIR = REPO_ROOT / "data"  # frozen, redistributable inputs (species.csv, tree.nwk)

for _d in (CACHE_DIR, ACT_DIR, RESULTS_DIR, LOGS_DIR, DATA_REPO_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    """Settings for one run (one model x one condition), loaded from YAML.

    Local: plain dataclass with defaults matching the preregistration.
    Global: passed to every phase; its ``name`` keys activation and result files.
    """

    name: str  # run id, e.g. "qwen35_2b"
    model_id: str = "Qwen/Qwen3.5-2B"  # Hugging Face model id
    revision: str | None = None  # pinned commit sha for reproducibility
    random_init: bool = False  # True -> same architecture with random weights (control)
    chat_template: bool = True  # wrap prompts in the model's chat template (user turn)
    prepend_bos: bool = False  # Pythia: prepend BOS manually (it does not add one)
    batch_size: int = 32  # prompts per forward pass
    dtype: str = "bfloat16"  # model weights dtype
    seed: int = 0  # global seed (random init, permutations, sampling)
    species_csv: str = "data/species.csv"  # frozen species list
    name_set: str = "species"  # "species" (main) | "conflict" | "days" | "months"
    n_perm: int = 1000  # permutations for max-T tests
    n_boot: int = 1000  # clade-block bootstrap resamples
    k_per_node: int = 200  # triplets sampled per decision node
    extra: dict = field(default_factory=dict)  # free-form stage-specific options


def load_config(path: str | Path) -> RunConfig:
    """Read a YAML run config into a :class:`RunConfig`.

    Unknown keys go into ``extra`` so stage-specific options do not break loading.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f for f in RunConfig.__dataclass_fields__ if f != "extra"}
    extra = {k: v for k, v in raw.items() if k not in known}
    return RunConfig(**{k: v for k, v in raw.items() if k in known}, extra=extra)


def setup_logging(run_name: str, level: int = logging.INFO) -> logging.Logger:
    """Log to ``logs/<timestamp>_<run_name>.log`` and to stdout, unbuffered.

    Local: idempotent; re-calling replaces handlers so logs are not duplicated.
    Global: all modules use ``logging.getLogger("phylogeo.<module>")`` children, so
    one call at the top of :mod:`phylogeo.run` captures everything.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{stamp}_{run_name}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger("phylogeo")
    root.setLevel(level)
    root.handlers.clear()  # avoid duplicate handlers when called twice
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    root.propagate = False
    root.info("logging to %s", log_path)
    return root
