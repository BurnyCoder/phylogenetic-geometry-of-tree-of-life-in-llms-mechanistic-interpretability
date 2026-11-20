"""Baseline and oracle distance matrices (preregistration section 4.4).

Global context: every metric computed on the LLM is also computed on these, so a
claim like "the model encodes phylogeny beyond surface form / co-occurrence /
folk classes" is a comparison against a concrete alternative:
  char3   lexical: Jaccard distance of character 3-gram sets of the names
  glove   co-occurrence: GloVe 6B-300d (Wikipedia 2014 + Gigaword 5; Pennington
          et al. 2014, https://nlp.stanford.edu/projects/glove/), mean over words,
          loaded from the sentence-transformers mirror
          https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d
  class   folk-class oracle: 0 within stratum, 1 between
  rank    Linnaean oracle: 5 - number of shared ranks (phylum..genus)
"""

from __future__ import annotations

import json
import logging

import numpy as np

log = logging.getLogger("phylogeo.baselines")


def char_ngrams(name: str, n: int = 3) -> set[str]:
    """Set of character n-grams of the lowercased name padded with spaces."""
    s = f" {name.lower()} "
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_ngram_distance(names: list[str], n: int = 3) -> np.ndarray:
    """[n, n] Jaccard distance between character n-gram sets."""
    grams = [char_ngrams(x, n) for x in names]
    k = len(names)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            inter = len(grams[i] & grams[j])
            union = len(grams[i] | grams[j])
            D[i, j] = D[j, i] = 1.0 - inter / union
    return D


def class_oracle(strata) -> np.ndarray:
    """0 if same stratum else 1 (the "folk categories only" geometry)."""
    s = np.asarray(strata)
    return (s[:, None] != s[None, :]).astype(float)


def rank_oracle(ranks: dict[str, np.ndarray], order=("phylum", "class", "order", "family", "genus")) -> np.ndarray:
    """5 - number of ranks shared (empty rank names never count as shared)."""
    n = len(next(iter(ranks.values())))
    shared = np.zeros((n, n))
    for r in order:
        v = np.asarray(ranks[r], dtype=object)
        same = (v[:, None] == v[None, :]) & (v[:, None] != "")
        shared += same
    D = len(order) - shared
    np.fill_diagonal(D, 0.0)
    return D


def load_glove():
    """Return (vocab dict word->row, embedding matrix [V, 300]) from the HF mirror."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    repo = "sentence-transformers/average_word_embeddings_glove.6B.300d"
    vocab = json.load(open(hf_hub_download(repo, "0_WordEmbeddings/whitespacetokenizer_config.json"),
                           encoding="utf-8"))["vocab"]
    W = load_file(hf_hub_download(repo, "0_WordEmbeddings/model.safetensors"))["emb_layer.weight"]
    return {w: i for i, w in enumerate(vocab)}, W


def glove_vectors(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean GloVe vector of each name's words (lowercased, hyphens/apostrophes split).

    Returns (vectors [n, 300], boolean mask of names with at least one known word).
    Names with no known word get the mean vector of all names as a placeholder;
    callers must exclude triplets involving them (``mask``), since their distances
    are float noise around the centre (review finding, minor).
    """
    vocab, W = load_glove()
    vecs, ok = [], []
    for name in names:
        words = name.lower().replace("-", " ").replace("'s", " ").replace("'", " ").split()
        rows = [vocab[w] for w in words if w in vocab]
        ok.append(bool(rows))
        vecs.append(W[rows].mean(0) if rows else np.full(W.shape[1], np.nan))
    V = np.array(vecs)
    fill = np.nanmean(V, axis=0)
    V[~np.array(ok)] = fill
    log.info("GloVe coverage: %d / %d names", int(np.sum(ok)), len(names))
    return V, np.array(ok)


def scalar_distance(x: np.ndarray) -> np.ndarray:
    """|x_i - x_j| for a 1-D nuisance variable (token count, log popularity)."""
    x = np.asarray(x, dtype=float)
    return np.abs(x[:, None] - x[None, :])
