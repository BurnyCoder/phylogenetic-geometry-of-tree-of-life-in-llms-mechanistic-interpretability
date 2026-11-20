"""Extract residual-stream representations of names from a Hugging Face causal LM.

Global context: the "extract" phase. For every (name, template) pair it runs one
forward pass (batch size 1: exact and reproducible; bf16 batched passes differ
by up to ~2.5% relative norm from unbatched ones on Qwen3.5-2B, see
reports/experiments/learnings.md) and stores, for every layer
(``hidden_states[0]`` = embedding output ... ``hidden_states[L]``), the vector at
the last name token and the mean over name tokens. Output arrays have shape
[n_names, n_templates, n_layers + 1, d] in float32 and are saved to
``DATA_DIR/activations/<run>/``.
Docs: https://huggingface.co/docs/transformers/main_classes/output (hidden_states).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from phylogeo.prompts import name_token_indices, render

log = logging.getLogger("phylogeo.extract")


def load_model(model_id: str, revision: str | None = None, random_init: bool = False,
               dtype: str = "bfloat16", seed: int = 0, device: str = "cuda"):
    """Load tokenizer + causal LM in ``dtype`` on ``device`` (whole model on GPU).

    Local: for multimodal checkpoints (Qwen3.5) ``AutoModelForCausalLM`` loads
    the text decoder only (verified identical hidden states to the full model).
    ``random_init=True`` builds the same architecture with freshly initialized
    weights (``from_config`` under a fixed seed): the untrained control.
    """
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    td = getattr(torch, dtype)
    if random_init:
        cfg = AutoConfig.from_pretrained(model_id, revision=revision)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg, dtype=td)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=td,
                                                     device_map=device)
    model.eval()
    log.info("loaded %s (random_init=%s) on %s; GPU mem %.2f GB", model_id, random_init, device,
             torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)
    return tok, model


@torch.no_grad()
def extract_names(tok, model, names: list[str], templates: list[str], chat: bool,
                  prepend_bos: bool = False, device: str = "cuda", log_every: int = 50):
    """Return (last_tok, mean_tok) arrays [n_names, n_templates, n_layers + 1, d] (float32).

    Local: one forward pass per prompt with ``output_hidden_states=True``; the
    name's tokens are located by character offsets; prompts and the tokens used
    are logged for the first name so the readout can be inspected in the logs.
    """
    n_layers = model.config.get_text_config().num_hidden_layers if hasattr(model.config, "get_text_config") \
        else model.config.num_hidden_layers
    d = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") \
        else model.config.hidden_size
    last = np.zeros((len(names), len(templates), n_layers + 1, d), np.float32)
    mean = np.zeros_like(last)
    t0 = time.time()
    for i, name in enumerate(names):
        for j, tpl in enumerate(templates):
            rp = render(tpl, name, tok, chat=chat)
            text = rp.text
            shift = 0
            if prepend_bos and tok.bos_token:
                text, shift = tok.bos_token + text, len(tok.bos_token)
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            offs = enc.pop("offset_mapping")[0].tolist()
            idx = name_token_indices(offs, rp.start + shift, rp.end + shift)
            out = model(input_ids=enc["input_ids"].to(device), output_hidden_states=True)
            hs = torch.stack(out.hidden_states)[:, 0]  # [L+1, T, d]
            last[i, j] = hs[:, idx[-1]].float().cpu().numpy()
            mean[i, j] = hs[:, idx].float().mean(1).cpu().numpy()
            if i == 0:
                toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
                log.info("prompt %d: %r | name tokens: %s", j, text, [toks[k] for k in idx])
        if (i + 1) % log_every == 0 or i + 1 == len(names):
            log.info("extracted %d / %d names (%.1fs)", i + 1, len(names), time.time() - t0)
    return last, mean


def save_activations(out_dir: Path, last: np.ndarray, mean: np.ndarray, names: list[str]) -> None:
    """Save arrays as float32 (float16 could overflow on LLM "massive activations")."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "last.npy", last.astype(np.float32))
    np.save(out_dir / "mean.npy", mean.astype(np.float32))
    (out_dir / "names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    log.info("saved activations %s to %s", last.shape, out_dir)


def load_activations(out_dir: Path, readout: str = "last") -> np.ndarray:
    """Load a saved activation array as float32 [n_names, n_templates, n_layers + 1, d]."""
    return np.load(out_dir / f"{readout}.npy").astype(np.float32)
