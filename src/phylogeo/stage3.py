"""Stage 3: behavioral judgments and causal tests of the phylogeny subspace.

Global context (preregistration: causal hypotheses C-abl, C-swap; exploratory
behavioral analyses; deviation 19). Steps:
  1. Behavior: relatedness judgments (A/B, both orders) on the conflict triplets
     and on within-stratum species triplets; class-word accuracy for all species.
  2. Causal: fit the flat-probe subspace on half of the families at the stage-1b
     layer; on the other (held-out) half, mean-ablate it at the name tokens and
     measure the drop in log P(correct class word), vs 99 variance-matched random
     subspaces (rank p-value, floor 0.01); interchange it between species of
     different classes and measure how often the answer switches to the source's
     class; off-target effect = change in token-weighted next-token loss on 20
     neutral sentences under the same ablation applied at every position.
Candidate class words are scored by their full-sequence log-probability in one
right-padded batch (first tokens collide for Qwen3.5: "Mammal"/"Mollusc").
"""

from __future__ import annotations

import contextlib
import json
import logging

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

from phylogeo.causal import residual_edit
from phylogeo.config import RESULTS_DIR, RunConfig

log = logging.getLogger("phylogeo.stage3")

NEUTRAL = [
    "The meeting was moved to Thursday because the room was unavailable.",
    "She opened the window and listened to the rain on the roof.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Our train arrived at the station twenty minutes late.",
    "He spent the afternoon repairing the old bicycle in the garage.",
    "The library extended its opening hours during the exam period.",
    "They painted the kitchen walls a pale shade of yellow.",
    "The software update fixed several bugs in the login screen.",
    "Prices at the market rose slightly over the summer.",
    "The children built a sandcastle near the edge of the water.",
    "A new bridge will connect the two halves of the city next year.",
    "The orchestra rehearsed the symphony for three hours.",
    "My phone battery drains quickly when the screen is bright.",
    "The museum displays paintings from the eighteenth century.",
    "We planted tomatoes and peppers in the small garden.",
    "The committee approved the budget after a long discussion.",
    "Snow closed the mountain pass for most of the week.",
    "He wrote the letter by hand and mailed it the next morning.",
    "The factory produces parts for washing machines.",
    "Fresh bread was on sale at the bakery on the corner.",
]


def _candidate_token_ids(tok, words: list[str]) -> list[list[int]]:
    """Token ids of each (capitalized) candidate class word.

    First tokens are NOT unique for Qwen3.5 ("Mammal" and "Mollusc" both start
    with "M"), so candidates are scored by their full-sequence log-probability.
    """
    return [tok(w.capitalize(), add_special_tokens=False)["input_ids"] for w in words]


@torch.no_grad()
def _class_logp(model, ids: torch.Tensor, cand: list[list[int]], pad_id: int) -> np.ndarray:
    """log-softmax over candidates of log P(candidate | prompt), one batched forward pass.

    Rows are prompt + candidate, right-padded (padding after the content cannot
    affect earlier positions in a causal model). Interventions registered as hooks
    act on the shared prompt positions in every row.
    """
    T = ids.shape[1]
    L = T + max(len(c) for c in cand)
    batch = torch.full((len(cand), L), pad_id, dtype=ids.dtype, device=ids.device)
    attn = torch.zeros_like(batch)
    for k, c in enumerate(cand):
        batch[k, :T] = ids[0]
        batch[k, T:T + len(c)] = torch.tensor(c, device=ids.device)
        attn[k, :T + len(c)] = 1
    lp = torch.log_softmax(model(input_ids=batch, attention_mask=attn).logits.float(), -1)
    s = torch.stack([sum(lp[k, T - 1 + j, t] for j, t in enumerate(c)) for k, c in enumerate(cand)])
    return (s - torch.logsumexp(s, 0)).cpu().numpy()


@torch.no_grad()
def _hidden_at(model, ids: torch.Tensor, layer: int, pos: list[int]) -> np.ndarray:
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0]
    return hs[pos].float().cpu().numpy()


@torch.no_grad()
def _neutral_loss(tok, model, layer: int | None = None, edit=None, device="cuda") -> float:
    """Token-weighted next-token loss over all NEUTRAL sentences.

    ``edit(n_tokens)`` (optional) returns a hidden-state function that is applied
    at ``layer`` for the whole sentence. The same function computes the no-edit
    and the edited losses, so they are comparable (deviation 19).
    """
    tot, n = 0.0, 0
    for s in NEUTRAL:
        ids = tok(s, return_tensors="pt")["input_ids"].to(device)
        ctx = residual_edit(model, layer, edit(ids.shape[1])) if edit else contextlib.nullcontext()
        with ctx:
            loss = float(model(input_ids=ids, labels=ids).loss)  # mean over ids.shape[1] - 1 predicted tokens
        tot += loss * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return tot / n


def _rank_p(x: float, null) -> float:
    """One-sided rank p-value of ``x`` among the null draws: (1 + #{null >= x}) / (n + 1)."""
    null = np.asarray(null)
    return float((1 + np.sum(null >= x)) / (len(null) + 1))


def _behavior(cfg, tok, model, df, tree, names, strata, Vl, has, rng, res) -> None:
    """Exploratory behavioral analyses (conflict triplets, species triplets, class accuracy)."""
    from phylogeo.behavior import CANDIDATES, CLASS_WORD, class_probs, generate_answer, triplet_judgment
    from phylogeo.geometry import cosine_distance
    from phylogeo.stats.triplets import distinct_mask, sample_triplets, within_group_mask

    # --- 1a. Behavior on conflict triplets ---------------------------------
    conf = json.loads((RESULTS_DIR / cfg.name / "conflict" / "conflict.json").read_text())
    rows = []
    for t in conf["triplets"]:
        if not t.get("resolved"):
            continue
        lo = triplet_judgment(tok, model, t["anchor"], t["tree_closer"], t["other"])
        rows.append({"id": t["id"], "anchor": t["anchor"], "tree_closer": t["tree_closer"], "other": t["other"],
                     "log_odds_tree_closer": lo})
    k = sum(r["log_odds_tree_closer"] > 0 for r in rows)
    res["behavior_conflict"] = {"n": len(rows), "chose_tree_closer": int(k), "accuracy": k / len(rows),
                                "binom_p": float(binomtest(k, len(rows), 0.5).pvalue), "triplets": rows}
    for r in rows[:5]:  # log a few free-text answers for inspection
        ans = generate_answer(tok, model, f"Which animal is more closely related to the {r['anchor']} in "
                                          f"evolutionary terms: the {r['tree_closer']} or the {r['other']}?", 40)
        log.info("generated | %s vs %s for %s -> %r", r["tree_closer"], r["other"], r["anchor"], ans)

    # --- 1b. Behavior on within-stratum species triplets --------------------
    trip = sample_triplets(tree, cfg.k_per_node, np.random.default_rng(cfg.seed))
    m = within_group_mask(trip, strata) & distinct_mask(trip, df["head_noun"].to_numpy())
    idx = rng.choice(np.where(m)[0], size=min(int(cfg.extra.get("n_behav_triplets", 300)), int(m.sum())), replace=False)
    beh = np.array([triplet_judgment(tok, model, names[trip.a[i]], names[trip.b[i]], names[trip.c[i]]) for i in idx])
    D = cosine_distance(Vl)
    rep = np.array([D[trip.a[i], trip.b[i]] < D[trip.a[i], trip.c[i]] for i in idx])
    res["behavior_species_triplets"] = {
        "n": int(len(idx)), "behavioral_accuracy": float((beh > 0).mean()),
        "binom_p": float(binomtest(int((beh > 0).sum()), len(idx), 0.5).pvalue),
        "representation_pairwise_accuracy_same_triplets": float(rep.mean()),
        "agreement_behavior_vs_representation": float(((beh > 0) == rep).mean())}
    log.info("behavior species triplets: %s", res["behavior_species_triplets"])

    # --- 1c. Class-word accuracy (full sequence scoring) --------------------
    probs = {i: class_probs(tok, model, names[i]) for i in np.where(has)[0]}
    correct = np.array([CANDIDATES[int(np.argmax(probs[i]))] == CLASS_WORD[strata[i]] for i in probs])
    res["class_accuracy"] = {"n": int(len(probs)), "accuracy": float(correct.mean()),
                             "by_stratum": {s: float(np.mean([CANDIDATES[int(np.argmax(probs[i]))] == CLASS_WORD[s]
                                                              for i in probs if strata[i] == s]))
                                            for s in sorted(set(strata[has]))}}
    log.info("class accuracy: %s", res["class_accuracy"])


def run_stage3(cfg: RunConfig, load_species, load_conflict, act_dir, out_dir, write_json,
               causal_only: bool = False) -> dict:
    """Run stage 3. ``causal_only`` keeps the stored behavioral results and recomputes only
    the causal tests; earlier causal results are kept under ``C_abl_v1`` / ``C_swap_v1``."""
    from phylogeo.behavior import CANDIDATES, CLASS_QUESTION, CLASS_WORD, chat_prefix
    from phylogeo.analysis import holm
    from phylogeo.causal import ablate_fn, name_positions, probe_subspace, random_subspaces, swap_fn
    from phylogeo.extract import load_activations, load_model
    from phylogeo.geometry import species_vectors
    from phylogeo.phases import check_names
    from phylogeo.probes.flat import family_folds

    rng = np.random.default_rng(cfg.seed)
    df, tree = load_species()
    names = df["common_name"].tolist()
    strata = df["stratum"].to_numpy()
    fam = np.where(df["family"] != "", df["family"], df["genus"]).astype(object)
    s1b = json.loads((RESULTS_DIR / cfg.name / "stage1b" / "results.json").read_text())
    layer = int(cfg.extra.get("stage3_layer", s1b["H1"]["modal_layer"]))
    layer = max(layer, 1)  # interventions need a decoder layer (layer 0 = embeddings)
    od = out_dir(cfg.name, "stage3")
    check_names(cfg.name, "species")
    Vl = species_vectors(load_activations(act_dir(cfg.name, "species"), "last"))[layer]
    has = np.array([s in CLASS_WORD for s in strata])  # species whose stratum has a class word
    tok, model = load_model(cfg.model_id, cfg.revision, False, cfg.dtype, cfg.seed)

    if causal_only:
        res = json.loads((od / "results.json").read_text())
        for key in ("C_abl", "C_swap"):
            if key in res and f"{key}_v1" not in res:
                res[f"{key}_v1"] = res.pop(key)  # first run, kept for transparency (deviation 19)
    else:
        res = {"layer": layer}
        _behavior(cfg, tok, model, df, tree, names, strata, Vl, has, rng, res)
        write_json(od / "results.json", res)

    # --- 2. Causal tests on held-out families -------------------------------
    n_rand = int(cfg.extra.get("n_random_subspaces", 99))  # rank-p floor 1/(n+1) = 0.01 (deviation 19)
    folds = family_folds(fam, 2, np.random.default_rng(cfg.seed))
    test = folds[0] & has
    train = ~folds[0]
    U, var = probe_subspace(Vl, tree.path_length(), train, k=10, seed=cfg.seed)
    Urs = random_subspaces(Vl[train], 10, var, n_rand, np.random.default_rng([cfg.seed, 1]))
    cand_ids = _candidate_token_ids(tok, CANDIDATES)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompts, pos = {}, {}
    for i in np.where(has)[0]:
        text = chat_prefix(tok, CLASS_QUESTION.format(name=names[i]))
        prompts[i] = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        pos[i] = name_positions(tok, text, names[i])
    # Mean activation at the name's last token in the class prompt (training species).
    H = {i: _hidden_at(model, prompts[i], layer, [pos[i][-1]])[0] for i in np.where(has)[0]}
    mu = np.mean([H[i] for i in np.where(train & has)[0]], axis=0)
    cw = {i: CANDIDATES.index(CLASS_WORD[strata[i]]) for i in np.where(has)[0]}
    te = np.where(test)[0]
    base_lp = {i: _class_logp(model, prompts[i], cand_ids, pad_id) for i in te}  # no-edit scores, computed once

    def drop(Ux):
        """Per held-out species: log P(correct) without edit minus with the subspace mean-ablated."""
        out = []
        for i in te:
            with residual_edit(model, layer, ablate_fn(Ux, mu, pos[i])):
                ab = _class_logp(model, prompts[i], cand_ids, pad_id)[cw[i]]
            out.append(base_lp[i][cw[i]] - ab)
        return np.array(out)

    d_probe = drop(U)
    d_rand = np.array([drop(Ur) for Ur in Urs])
    w = wilcoxon(d_probe, d_rand.mean(0), alternative="greater")

    def off_target(Ux=None) -> float:
        edit = None if Ux is None else (lambda n: ablate_fn(Ux, mu, list(range(n))))
        return _neutral_loss(tok, model, layer, edit)

    res["C_abl"] = {"n_test_species": int(test.sum()), "n_random_subspaces": n_rand, "probe_var_captured": var,
                    "mean_base_logp_correct": float(np.mean([base_lp[i][cw[i]] for i in te])),
                    "mean_drop_logp_probe": float(d_probe.mean()), "mean_drop_logp_random": float(d_rand.mean()),
                    "random_mean_drops": d_rand.mean(1), "p_rank_vs_random": _rank_p(d_probe.mean(), d_rand.mean(1)),
                    "wilcoxon_p_probe_gt_random": float(w.pvalue), "neutral_loss_base": off_target(),
                    "neutral_loss_probe_ablated": off_target(U),
                    "neutral_loss_random_ablated": float(np.mean([off_target(Ur) for Ur in Urs[:10]]))}
    log.info("C-abl: %s", {k_: v for k_, v in res["C_abl"].items() if k_ != "random_mean_drops"})

    # Interchange: pairs (source A, target B) of held-out species with different class words.
    rng_pairs = np.random.default_rng([cfg.seed, 2])
    pairs = []
    while len(pairs) < int(cfg.extra.get("n_swap_pairs", 200)):
        a, b = rng_pairs.choice(te, 2, replace=False)
        if cw[a] != cw[b]:
            pairs.append((a, b))

    def switch_rate(Ux):
        """Fraction of pairs where writing A's subspace coordinates into B's name token
        makes A's class word the top candidate for B."""
        sw = 0
        for a, b in pairs:
            with residual_edit(model, layer, swap_fn(Ux, H[a], pos[b][-1])):
                lp = _class_logp(model, prompts[b], cand_ids, pad_id)
            sw += int(np.argmax(lp) == cw[a])
        return sw / len(pairs)

    base_sw = np.mean([int(np.argmax(base_lp[b]) == cw[a]) for a, b in pairs])
    sr_probe = switch_rate(U)
    sr_rand = [switch_rate(Ur) for Ur in Urs]
    res["C_swap"] = {"n_pairs": len(pairs), "n_random_subspaces": n_rand, "switch_rate_no_edit": float(base_sw),
                     "switch_rate_probe": sr_probe, "switch_rate_random": sr_rand,
                     "p_rank_vs_random": _rank_p(sr_probe, sr_rand)}
    res["causal_holm"] = holm({"C_abl": res["C_abl"]["p_rank_vs_random"], "C_swap": res["C_swap"]["p_rank_vs_random"]})
    log.info("C-swap: %s", {k_: v for k_, v in res["C_swap"].items() if k_ != "switch_rate_random"})
    log.info("causal Holm (rank p): %s", res["causal_holm"])
    write_json(od / "results.json", res)
    del model
    torch.cuda.empty_cache()
    return res
