"""Tests for residual-stream interventions on a tiny random Qwen3.5 model (CPU)."""

import numpy as np
import pytest
import torch

from phylogeo.causal import ablate_fn, orthonormal, random_subspaces, residual_edit, swap_fn


@pytest.fixture(scope="module")
def tiny_qwen():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", revision="15852e8c16360a2fea060d615a32b45270f8a8fc")
    tc = cfg.get_text_config()
    tc.num_hidden_layers = 4
    tc.layer_types = tc.layer_types[:4]
    tc.hidden_size = 64
    tc.intermediate_size = 128
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 16
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.linear_key_head_dim = 16
    tc.linear_value_head_dim = 16
    tc.vocab_size = 1000
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(tc).float().eval()
    return m


def _hidden(m, ids, layer):
    with torch.no_grad():
        return m(input_ids=ids, output_hidden_states=True).hidden_states[layer]


def _edited_output(m, ids, layer):
    """Output of decoder layer ``layer - 1`` *after* all hooks (captured by a later hook).

    transformers records ``hidden_states`` with its own earlier hook, so it shows the
    pre-edit value; the edited value is what flows to the next layer.
    """
    box = {}
    h = m.model.layers[layer - 1].register_forward_hook(lambda mod, i, o: box.setdefault("h", o))
    with torch.no_grad():
        m(input_ids=ids)
    h.remove()
    return box["h"]


def test_identity_hook_changes_nothing(tiny_qwen):
    ids = torch.randint(0, 1000, (1, 7))
    base = _hidden(tiny_qwen, ids, 2)
    with residual_edit(tiny_qwen, 2, lambda h: h):
        same = _hidden(tiny_qwen, ids, 2)
    assert torch.allclose(base, same)


def test_ablation_removes_subspace_component(tiny_qwen):
    ids = torch.randint(0, 1000, (1, 7))
    rng = np.random.default_rng(0)
    U = orthonormal(rng.normal(size=(3, 64)))
    mu = rng.normal(size=64)
    with residual_edit(tiny_qwen, 2, ablate_fn(U, mu, [3, 4])):
        h = _edited_output(tiny_qwen, ids, 2)[0].numpy()
        nxt = _hidden(tiny_qwen, ids, 3)
    assert not torch.allclose(nxt, _hidden(tiny_qwen, ids, 3))  # edit propagates downstream
    # At edited positions the subspace coordinates equal those of mu.
    assert np.allclose(h[[3, 4]] @ U, np.tile(mu @ U, (2, 1)), atol=1e-4)
    base = _hidden(tiny_qwen, ids, 2)[0].numpy()
    assert np.allclose(h[:3], base[:3], atol=1e-5)  # earlier positions untouched


def test_swap_sets_source_coordinates(tiny_qwen):
    ids = torch.randint(0, 1000, (1, 6))
    rng = np.random.default_rng(1)
    U = orthonormal(rng.normal(size=(2, 64)))
    src = rng.normal(size=64)
    with residual_edit(tiny_qwen, 3, swap_fn(U, src, 5)):
        h = _edited_output(tiny_qwen, ids, 3)[0].numpy()
    assert np.allclose(h[5] @ U, src @ U, atol=1e-4)


def test_random_subspaces_variance_matched():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(200, 50)) * np.linspace(5, 0.1, 50)
    Vc = V - V.mean(0)
    target = 0.3
    Us = random_subspaces(V, 3, target, 20, rng)
    got = np.mean([((Vc @ U) ** 2).sum() / (Vc**2).sum() for U in Us])
    assert abs(got - target) < 0.1
    assert all(np.allclose(U.T @ U, np.eye(3), atol=1e-8) for U in Us)


def test_batched_candidate_scoring_matches_sequential(tiny_qwen):
    """stage3._class_logp (batched, right-padded) equals per-candidate sequence log-probs."""
    from phylogeo.stage3 import _class_logp

    torch.manual_seed(0)
    prefix = torch.randint(0, 1000, (1, 6))
    cands = [[5, 7], [9], [11, 12, 13]]
    got = _class_logp(tiny_qwen, prefix, cands, pad_id=0)
    seq = []
    with torch.no_grad():
        for c in cands:
            ids = torch.cat([prefix[0], torch.tensor(c)])[None]
            lp = torch.log_softmax(tiny_qwen(input_ids=ids).logits[0].float(), -1)
            seq.append(sum(lp[5 + j, t] for j, t in enumerate(c)))
    s = torch.stack(seq)
    want = (s - torch.logsumexp(s, 0)).numpy()
    assert np.allclose(got, want, atol=1e-4)


def test_rank_p_floor_and_ties():
    """Rank p-value counts ties against the probe and cannot go below 1/(n+1)."""
    from phylogeo.stage3 import _rank_p

    assert _rank_p(5.0, [1, 2, 3]) == 0.25  # above all 3 nulls -> floor 1/4
    assert _rank_p(2.0, [1, 2, 3]) == 0.75  # tie counts as >=
    assert _rank_p(0.0, [1, 2, 3]) == 1.0


def test_neutral_loss_base_and_edit_use_same_computation(tiny_qwen):
    """Deviation 19: an identity edit reproduces the no-edit loss exactly; an ablation changes it."""
    from phylogeo.stage3 import _neutral_loss

    def tok(s, return_tensors="pt"):  # toy tokenizer: characters -> ids in the tiny vocab
        return {"input_ids": torch.tensor([[ord(c) % 1000 for c in s[:12]]])}

    base = _neutral_loss(tok, tiny_qwen, device="cpu")
    same = _neutral_loss(tok, tiny_qwen, 2, lambda n: (lambda h: h), device="cpu")
    assert abs(base - same) < 1e-6
    U = orthonormal(np.random.default_rng(0).normal(size=(3, 64)))
    changed = _neutral_loss(tok, tiny_qwen, 2, lambda n: ablate_fn(U, np.ones(64), list(range(n))), device="cpu")
    assert abs(base - changed) > 1e-6
