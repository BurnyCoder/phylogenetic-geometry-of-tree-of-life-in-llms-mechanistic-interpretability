"""Figures: interactive 3D PCA (Plotly, layer slider), 2D PCA panels and grids (matplotlib).

Global context: stage 1a ("do they look like a tree?") is answered first by eye,
so these plots are the primary artifact of that stage. Uncolored variants exist
for the preregistered anti-bias step (describe the shape before seeing clades).
Plotly docs: https://plotly.com/python/sliders/ ; matplotlib: https://matplotlib.org/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering to files
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from phylogeo.data.tree import RefTree
from phylogeo.viz.pca import STRATUM_STYLE, overlay_positions

GRAY = "#d6d5d0"  # recessive context color for non-highlighted points


def _edge_xyz(tree: RefTree, coords: np.ndarray) -> tuple[list, list, list]:
    """Line-segment coordinates (with None separators) of the tree overlay in PC space."""
    pos = overlay_positions(tree, coords)
    xs, ys, zs = [], [], []
    for v in range(tree.N):
        if v == tree.root:
            continue
        p = tree.parent[v]
        xs += [pos[p, 0], pos[v, 0], None]
        ys += [pos[p, 1], pos[v, 1], None]
        zs += [pos[p, 2], pos[v, 2], None]
    return xs, ys, zs


def pca_3d_html(coords_by_layer: list[np.ndarray], evr_by_layer: list[np.ndarray], names: list[str],
                strata: np.ndarray, tree: RefTree | None, path: Path, title: str,
                template_coords_by_layer: list[np.ndarray] | None = None, colored: bool = True,
                start_layer: int | None = None) -> None:
    """Interactive 3D scatter of the top-3 PCs with one slider step per layer.

    Traces per frame: one per stratum (big dots, hover = name), the tree overlay
    (thin gray lines, if ``tree``), and the per-template points (small dots,
    hidden by default; toggle in the legend).
    """
    strata = np.asarray(strata)
    groups = list(STRATUM_STYLE) if colored else ["all"]

    def frame_traces(l: int):
        c = coords_by_layer[l]
        tr = []
        for g in groups:
            m = (strata == g) if colored else np.ones(len(strata), bool)
            if not m.any():
                continue
            tr.append(go.Scatter3d(x=c[m, 0], y=c[m, 1], z=c[m, 2], mode="markers", name=g,
                                   text=[names[i] for i in np.where(m)[0]], hoverinfo="text+name",
                                   marker=dict(size=4, color=STRATUM_STYLE[g][0] if colored else "#555",
                                               symbol=STRATUM_STYLE[g][2] if colored else "circle")))
        if tree is not None:
            xs, ys, zs = _edge_xyz(tree, c)
            tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="reference tree",
                                   line=dict(color="rgba(80,80,80,0.35)", width=1), hoverinfo="skip"))
        if template_coords_by_layer is not None:
            t = template_coords_by_layer[l]
            tr.append(go.Scatter3d(x=t[:, 0], y=t[:, 1], z=t[:, 2], mode="markers", name="single prompts",
                                   marker=dict(size=1.5, color="rgba(0,0,0,0.25)"), hoverinfo="skip",
                                   visible="legendonly"))
        return tr

    L = len(coords_by_layer)
    l0 = start_layer if start_layer is not None else L // 2
    frames = [go.Frame(data=frame_traces(l), name=str(l)) for l in range(L)]
    fig = go.Figure(data=frame_traces(l0), frames=frames)
    steps = [dict(method="animate", label=str(l),
                  args=[[str(l)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for l in range(L)]
    ev = evr_by_layer[l0]
    fig.update_layout(
        title=f"{title} — top-3 PCA per layer (layer slider; hover for names)",
        sliders=[dict(active=l0, currentvalue=dict(prefix="layer "), steps=steps)],
        scene=dict(xaxis_title=f"PC1 ({ev[0]:.1%} at layer {l0})", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(itemsizing="constant"), height=800, template="plotly_white")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def pca_2d_png(coords: np.ndarray, evr: np.ndarray, strata: np.ndarray, tree: RefTree | None,
               path: Path, title: str, colored: bool = True) -> None:
    """Three 2D panels (PC1-2, PC2-3, PC1-3) with explained variance on the axes."""
    strata = np.asarray(strata)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    pos = overlay_positions(tree, coords) if (tree is not None and colored) else None
    for ax, (a, b) in zip(axes, [(0, 1), (1, 2), (0, 2)]):
        if pos is not None:
            for v in range(tree.N):
                if v != tree.root:
                    p = tree.parent[v]
                    ax.plot([pos[p, a], pos[v, a]], [pos[p, b], pos[v, b]], color="0.7", lw=0.4, zorder=1)
        if colored:
            for g, (col, mk, _) in STRATUM_STYLE.items():
                m = strata == g
                if m.any():
                    ax.scatter(coords[m, a], coords[m, b], s=14, color=col, marker=mk, label=g, zorder=2,
                               edgecolors="white", linewidths=0.3)
        else:
            ax.scatter(coords[:, a], coords[:, b], s=10, color="0.3", zorder=2)
        ax.set_xlabel(f"PC{a + 1} ({evr[a]:.1%} var)")
        ax.set_ylabel(f"PC{b + 1} ({evr[b]:.1%} var)")
    if colored:
        axes[-1].legend(fontsize=7, markerscale=1.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pca_layer_grid_png(coords_by_layer: list[np.ndarray], evr_by_layer: list[np.ndarray], strata: np.ndarray,
                       path: Path, title: str, colored: bool = True, dims=(0, 1)) -> None:
    """Grid of PC-a vs PC-b scatters, one panel per layer."""
    L = len(coords_by_layer)
    ncol = 5
    nrow = int(np.ceil(L / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
    strata = np.asarray(strata)
    a, b = dims
    for l, ax in enumerate(axes.ravel()):
        if l >= L:
            ax.axis("off")
            continue
        c = coords_by_layer[l]
        if colored:
            for g, (col, mk, _) in STRATUM_STYLE.items():
                m = strata == g
                if m.any():
                    ax.scatter(c[m, a], c[m, b], s=6, color=col, marker=mk, label=g, linewidths=0)
        else:
            ax.scatter(c[:, a], c[:, b], s=3, color="0.3")
        ev = evr_by_layer[l]
        ax.set_title(f"layer {l}  (PC{a+1} {ev[a]:.0%}, PC{b+1} {ev[b]:.0%})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    if colored:
        h, lab = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=7, fontsize=8, markerscale=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04 if colored else 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def circle_png(coords: np.ndarray, evr: np.ndarray, labels: list[str], path: Path, title: str, dims=(0, 1)) -> None:
    """Calibration plot for days/months: points labeled and connected in calendar order."""
    a, b = dims
    fig, ax = plt.subplots(figsize=(5, 5))
    cyc = np.r_[np.arange(len(labels)), 0]
    ax.plot(coords[cyc, a], coords[cyc, b], color="0.7", lw=1)
    ax.scatter(coords[:, a], coords[:, b], c=np.arange(len(labels)), cmap="twilight", s=40, zorder=3)
    for i, t in enumerate(labels):
        ax.annotate(t, (coords[i, a], coords[i, b]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC{a+1} ({evr[a]:.1%})"); ax.set_ylabel(f"PC{b+1} ({evr[b]:.1%})")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def line_by_layer_png(series: dict[str, np.ndarray], path: Path, title: str, ylabel: str,
                      hlines: dict[str, float] | None = None, bands: dict[str, tuple] | None = None) -> None:
    """Metric vs layer for several representations (optional reference lines / CI bands)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, y in series.items():
        ax.plot(np.arange(len(y)), y, marker="o", ms=3, label=name)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(np.arange(len(y)), lo, hi, alpha=0.2)
    for name, v in (hlines or {}).items():
        ax.axhline(v, ls="--", lw=1, color="0.4")
        ax.text(0, v, f" {name}", va="bottom", fontsize=8, color="0.3")
    ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pca_spotlight_png(coords: np.ndarray, evr: np.ndarray, strata: np.ndarray, path: Path, title: str,
                      dims=(0, 1)) -> None:
    """Small multiples: one panel per stratum, that stratum highlighted, all others gray.

    Identity is carried by the panel title (not by color), which satisfies the
    dataviz rule for >3 categories in a scatter (facet instead of more hues).
    """
    strata = np.asarray(strata)
    groups = [g for g in STRATUM_STYLE if (strata == g).any()]
    ncol = 5
    nrow = int(np.ceil(len(groups) / ncol))
    a, b = dims
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow), sharex=True, sharey=True)
    for ax, g in zip(axes.ravel(), groups):
        m = strata == g
        ax.scatter(coords[~m, a], coords[~m, b], s=4, color=GRAY, linewidths=0)
        ax.scatter(coords[m, a], coords[m, b], s=10, color=STRATUM_STYLE["mammals"][0], linewidths=0)
        ax.set_title(f"{g} (n={int(m.sum())})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.supxlabel(f"PC{a + 1} ({evr[a]:.1%} var)", fontsize=9)
    fig.supylabel(f"PC{b + 1} ({evr[b]:.1%} var)", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
