"""Index claim-bearing files and their evidence without using previous audits as evidence."""

import json
import subprocess
from pathlib import Path

from phylogeo.config import REPO_ROOT
from .statistics import save

TREE = "https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3"
REP = "https://huggingface.co/datasets/BurnyCoder/phylogenetic-geometry-activations"
PERM = "https://doi.org/10.1007/s11749-017-0571-1"
TRIP = "https://doi.org/10.1186/s13015-020-00175-0"
PROBE = "https://aclanthology.org/D19-1275/"
CAUSE = "https://arxiv.org/abs/2106.02997"
ECO = "https://esapubs.org/archive/ecol/E095/178/"
TIME = "https://doi.org/10.1093/molbev/msac174"
PYTHIA = "https://arxiv.org/abs/2304.01373"
GEO = "https://www.goodfire.com/research/phylogeny-manifold"
AUDIT = "reports/reviews/2026-11-26-scientific-audit.md"
ROOT = "results/scientific-verification/"


def sources_for(path):
    p = Path(path)
    if path.startswith("reports/reviews/") and "2026-11-26" not in path:
        return "Historical document, superseded conclusions", [AUDIT], []
    if "/audit/" in path or p.stem in ("audit", "test_audit"):
        return "Previous audit, excluded from evidentiary inputs", [AUDIT], []
    if path.startswith("site/assets/"):
        stage = p.parent.name
        prefix = "results/stage5/" if stage == "stage5" else f"results/qwen35_2b/{stage}/"
        return "Historical graphic mirrored on the site", [prefix + p.name], [REP]
    if "preregistration" in path or path.endswith(("plan.md", "planned-runs.md", "analysis-protocol.md")):
        return "Dated plan or protocol, with subsequent deviations", ["reports/preregistration.md", AUDIT], ["https://doi.org/10.1073/pnas.1708274114"]
    if path.startswith("data/"):
        return "Frozen species, stimulus or topology input", ["data/FREEZE.json", ROOT + "geometry.json"], [TREE, ECO] if "eco" in path else [TREE]
    if path.startswith("configs/"):
        return "Experiment settings", [path, "docs/pipeline.md"], [REP, PYTHIA]
    if ("stage5" in path or "scale" in p.stem or "dynamics" in path or "more-models" in path):
        return "Archived checkpoint observation or its display", [ROOT + "scale.json", "results/stage5/summary.json"], [PYTHIA, REP]
    if "conflict" in path or "stage3" in path or "causal" in path or "behavior" in path:
        return "Conflict and intervention observations or code", [ROOT + "outcomes.json", "results/qwen35_2b/stage3/results.json"], [TRIP, CAUSE]
    if any(w in path for w in ("habitat", "/eco", "heldout", "foraging")):
        return "Held-out trait measurement or its display", [ROOT + "outcomes.json", "data/heldout_eco.csv"], [ECO, "https://doi.org/10.1086/284325"]
    if any(w in path for w in ("stage2", "probe", "treeness", "stage4", "stage6", "power", "sensitivity")):
        return "Archived exploratory measurement, simulation or implementation", [ROOT + "outcomes.json", "results/RESULTS.md"], [PROBE, GEO]
    if "dating" in path or "timetree" in path:
        return "Age and topology analysis", [ROOT + "dating.json", "src/phylogeo/data/tree.py"], [TREE, TIME]
    if any(w in path for w in ("stage1", "triplet", "geometry", "mantel", "/viz", "/figures", "pca", "heatmap")):
        return "Tree geometry, exploratory display or statistical implementation", [ROOT + "geometry.json", "results/qwen35_2b/stage1a/checks.json"], [TRIP, PERM, REP]
    if path.startswith("results/") or path.startswith("reports/") or path in ("README.md", "site/index.html"):
        return "Research report, frozen statistic or publication mirror", [AUDIT, ROOT + "decisions.json"], [TREE, PERM, REP]
    if path.startswith(("src/", "tests/", "docs/")):
        return "Implementation, regression check or operating instructions", ["docs/pipeline.md", "uv.lock", AUDIT], [REP]
    return "Project metadata, dependency lock or deployment configuration", [path], []


def write_index():
    listed = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=REPO_ROOT)
    paths = {v.decode("utf-8") for v in listed.split(b"\0") if v} | {ROOT + "file-sources.json"}
    entries = {}
    for path in sorted(paths):
        if not (REPO_ROOT / path).is_file() and path != ROOT + "file-sources.json":
            continue
        kind, local, web = sources_for(path)
        entries[path] = {"role": kind, "local": local, "web": web}
    save("file-sources", entries)


def original_observations():
    """Verify that the records interpreted as historical still match the original public revision."""
    rev = "cc36b9134981497d6006359183261381c25177ac"
    patterns = ["data/*.csv", "data/*.nwk", "data/FREEZE.json", "results/power/*.json",
                "results/*/stage5/scale.json", "results/qwen35_2b/stage*/results.json",
                "results/qwen35_2b/stage1a/checks.json", "results/qwen35_2b/eco/results.json",
                "results/qwen35_2b/conflict/conflict.json", "results/stage5/summary.json"]
    # Git blob IDs hash the stored content. hash-object uses each tracked path's
    # newline rules so equivalent Windows and Linux checkouts resolve to the same ID.
    command = ["git", "ls-files", "-z", "--", *patterns]
    paths = sorted(p.decode() for p in subprocess.check_output(command, cwd=REPO_ROOT).split(b"\0") if p)
    dest = REPO_ROOT / "results/scientific-verification/original-observations.json"
    verified = {}
    if dest.exists():
        manifest = json.loads(dest.read_text(encoding="utf-8"))
        expected = manifest["git_blobs"]
    else:
        expected = {p: subprocess.check_output(["git", "rev-parse", f"{rev}:{p}"], cwd=REPO_ROOT, text=True).strip()
                    for p in paths}
    blobs = subprocess.run(["git", "hash-object", "--stdin-paths"], cwd=REPO_ROOT,
                           input="\n".join(paths) + "\n", text=True, capture_output=True, check=True).stdout.splitlines()
    for path, blob in zip(paths, blobs, strict=True):
        if blob != expected.get(path):
            raise ValueError(f"{path} differs from the original audit input at {rev}")
        verified[path] = blob
    if set(paths) != set(expected):
        raise ValueError("An original audit input is missing from this checkout")
    save("original-observations", {"revision": rev, "git_blobs": verified})
