"""Causal tests of the phylogeny subspace (stage 3): mean-ablation and interchange.

Global context (preregistration, causal hypotheses C-abl and C-swap): if the
model *uses* the subspace found by the stage-2 probe, removing it should hurt a
downstream taxonomic judgment more than removing a variance-matched random
subspace, and swapping it between two species should move the judgment toward
the source species' class. Interventions edit the residual stream at the output
of decoder layer L-1 (= ``hidden_states[L]``) at the name's token positions via a
PyTorch forward hook (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook),
the standard activation-patching recipe (e.g. Geiger et al. interchange
interventions, https://arxiv.org/abs/2106.02997).
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import torch
from sklearn.decomposition import PCA

log = logging.getLogger("phylogeo.causal")


def orthonormal(M: np.ndarray) -> np.ndarray:
    """Orthonormal basis (columns) of the row space of M [k, d] -> [d, k]."""
    q, _ = np.linalg.qr(M.T)
    return q


def probe_subspace(V: np.ndarray, D: np.ndarray, train: np.ndarray, k: int = 10, n_pca: int = 100,
                   steps: int = 2000, seed: int = 0) -> tuple[np.ndarray, float]:
    """Fit the flat probe on training species; return its subspace in residual space [d, k].

    The probe acts on standardized PCA features Z = (V - m) P^T / s, so the linear
    map in residual coordinates is W P / s; its row space is the subspace.
    Returns (U [d, k], fraction of species-vector variance it captures).
    """
    from phylogeo.probes.flat import train_probe

    pca = PCA(n_components=min(n_pca, train.sum() - 1), random_state=seed).fit(V[train])
    Z = pca.transform(V)
    s = Z[train].std()
    m = train_probe(Z[train] / s, D[np.ix_(train, train)] * (np.pi / D.max()), k=k, steps=steps, seed=seed)
    M = m.W.detach().numpy() @ pca.components_ / s
    U = orthonormal(M)
    Vc = V[train] - V[train].mean(0)  # same species set as the variance-matched random subspaces (deviation 20)
    return U, float(((Vc @ U) ** 2).sum() / (Vc**2).sum())


def random_subspaces(V: np.ndarray, k: int, target_var: float, n: int, rng: np.random.Generator,
                     max_pcs: int = 200) -> list[np.ndarray]:
    """Random k-dim subspaces inside the span of the top-m PCs of V, with m chosen so the
    captured variance matches ``target_var`` on average (variance-matched control)."""
    Vc = V - V.mean(0)
    _, S, Wt = np.linalg.svd(Vc, full_matrices=False)
    tot = (S**2).sum()

    def sample(m):
        G = rng.normal(size=(k, m))
        return orthonormal(G @ Wt[:m])

    best_m, best_gap = k, np.inf
    for m in range(k, min(max_pcs, len(S)) + 1, 2):
        v = np.mean([((Vc @ sample(m)) ** 2).sum() / tot for _ in range(10)])
        if abs(v - target_var) < best_gap:
            best_m, best_gap = m, abs(v - target_var)
    out = [sample(best_m) for _ in range(n)]
    log.info("random subspaces from top-%d PCs; mean captured variance %.4f (target %.4f)", best_m,
             np.mean([((Vc @ U) ** 2).sum() / tot for U in out]), target_var)
    return out


@contextlib.contextmanager
def residual_edit(model, layer: int, fn):
    """Apply ``fn(hidden [B, T, d]) -> hidden`` to the output of decoder layer ``layer - 1``.

    ``layer`` indexes ``hidden_states`` (0 = embeddings), so layer >= 1.
    """
    blocks = model.model.layers
    h = blocks[layer - 1].register_forward_hook(
        lambda mod, inp, out: fn(out) if torch.is_tensor(out) else (fn(out[0]),) + tuple(out[1:]))
    try:
        yield
    finally:
        h.remove()


def ablate_fn(U: np.ndarray, mu: np.ndarray, positions: list[int]):
    """Mean-ablation h' = h - U U^T (h - mu) at the given token positions."""
    Ut = torch.tensor(U, dtype=torch.float32)
    mut = torch.tensor(mu, dtype=torch.float32)

    def fn(h):
        x = h[:, positions].float()
        Uu, m = Ut.to(h.device), mut.to(h.device)
        x = x - ((x - m) @ Uu) @ Uu.T
        h = h.clone()
        h[:, positions] = x.to(h.dtype)
        return h

    return fn


def swap_fn(U: np.ndarray, src: np.ndarray, position: int):
    """Interchange: replace the subspace component at ``position`` with that of ``src`` [d]."""
    Ut = torch.tensor(U, dtype=torch.float32)
    st = torch.tensor(src, dtype=torch.float32)

    def fn(h):
        Uu, s = Ut.to(h.device), st.to(h.device)
        x = h[:, position].float()
        x = x - (x @ Uu) @ Uu.T + (s @ Uu) @ Uu.T
        h = h.clone()
        h[:, position] = x.to(h.dtype)
        return h

    return fn


def name_positions(tok, text: str, name: str) -> list[int]:
    """Token indices of the last occurrence of ``name`` in ``text`` (offset mapping)."""
    from phylogeo.prompts import name_token_indices

    s = text.rindex(name)
    offs = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return name_token_indices(offs, s, s + len(name))
