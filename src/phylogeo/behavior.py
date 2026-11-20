"""Behavioral probes of the chat model: relatedness judgments and class classification.

Global context (stage 3): compares what the model *says* with what its geometry
encodes, and provides the readout used by the causal interventions.
  * Relatedness triplets: "Which animal is more closely related to the X ...?
    A. the Y  B. the Z" scored by the next-token log-probabilities of "A" vs "B"
    after the assistant prefix (thinking disabled), averaged over both option
    orders to cancel position bias (a standard multiple-choice scoring recipe,
    e.g. Hendrycks et al. 2021 MMLU, https://arxiv.org/abs/2009.03300).
  * Class classification: "What kind of animal is the X? ..." scored by the
    sequence log-probability of each candidate class word; the softmax over
    candidates gives P(correct class).
Every prompt and score is logged (logs/) so model outputs can be inspected.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger("phylogeo.behavior")

# Stratum -> class word the model should answer (strata without one are skipped).
CLASS_WORD = {
    "mammals": "mammal", "birds": "bird", "reptiles": "reptile", "amphibians": "amphibian",
    "ray-finned fish": "fish", "cartilaginous fish": "fish", "insects": "insect", "arachnids": "arachnid",
    "crustaceans": "crustacean", "molluscs": "mollusc", "cnidarians": "cnidarian", "echinoderms": "echinoderm",
}
CANDIDATES = sorted(set(CLASS_WORD.values()))
CLASS_QUESTION = ("What kind of animal is the {name}? Choose one of: " + ", ".join(CANDIDATES)
                  + ". Answer with one word.")
TRIPLET_QUESTION = ("Which animal is more closely related to the {a} in evolutionary terms?\n"
                    "A. the {x}\nB. the {y}\nAnswer with A or B only.")


def chat_prefix(tok, user: str) -> str:
    """Chat-formatted prompt ending with the assistant turn start (thinking disabled)."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def next_token_logprobs(tok, model, text: str, device: str = "cuda", hook_ctx=None) -> torch.Tensor:
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    logits = model(input_ids=ids).logits[0, -1].float()
    return torch.log_softmax(logits, -1).cpu()


@torch.no_grad()
def sequence_logprob(tok, model, prefix: str, cont: str, device: str = "cuda") -> float:
    """log P(cont | prefix) summed over the continuation tokens."""
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    c = tok(prefix + cont, add_special_tokens=False)["input_ids"][len(p):]
    ids = torch.tensor([p + c], device=device)
    lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
    return float(sum(lp[len(p) - 1 + i, t] for i, t in enumerate(c)))


def triplet_judgment(tok, model, a: str, x: str, y: str) -> float:
    """Mean log-odds that the model picks x over y as closer to a (both option orders)."""
    ida, idb = tok("A", add_special_tokens=False)["input_ids"][0], tok("B", add_special_tokens=False)["input_ids"][0]
    s1 = next_token_logprobs(tok, model, chat_prefix(tok, TRIPLET_QUESTION.format(a=a, x=x, y=y)))
    s2 = next_token_logprobs(tok, model, chat_prefix(tok, TRIPLET_QUESTION.format(a=a, x=y, y=x)))
    lo = 0.5 * ((s1[ida] - s1[idb]) + (s2[idb] - s2[ida]))
    log.info("triplet | anchor=%s | x=%s | y=%s | log-odds(x)=%.3f (order1 A-B %.2f, order2 B-A %.2f)",
             a, x, y, float(lo), float(s1[ida] - s1[idb]), float(s2[idb] - s2[ida]))
    return float(lo)


def class_probs(tok, model, name: str) -> np.ndarray:
    """Softmax over candidate class words (capitalized and lowercase variants pooled)."""
    prefix = chat_prefix(tok, CLASS_QUESTION.format(name=name))
    scores = []
    for w in CANDIDATES:
        v = [sequence_logprob(tok, model, prefix, w.capitalize()), sequence_logprob(tok, model, prefix, w)]
        scores.append(np.logaddexp(*v))
    s = np.array(scores)
    return np.exp(s - np.logaddexp.reduce(s))


@torch.no_grad()
def generate_answer(tok, model, user: str, max_new_tokens: int = 20, device: str = "cuda") -> str:
    """Greedy generation (for logging/inspection of what the model actually says)."""
    ids = tok(chat_prefix(tok, user), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
