"""Tests for prompt rendering, name-token location, and (slow) real-model extraction."""

import numpy as np
import pytest

from phylogeo.prompts import TEMPLATES, name_token_indices, render


class _FakeChatTok:
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        return "<|im_start|>user\n" + msgs[0]["content"] + "<|im_end|>\n"


def test_render_span_raw_and_chat():
    for tpl in TEMPLATES:
        rp = render(tpl, "house sparrow")
        assert rp.text[rp.start : rp.end] == "house sparrow"
        rc = render(tpl, "house sparrow", _FakeChatTok(), chat=True)
        assert rc.text[rc.start : rc.end] == "house sparrow"
        assert rc.text.startswith("<|im_start|>user\n")


def test_name_token_indices_overlap():
    # "The house sparrow is" tokenized as [The][ house][ sp][arrow][ is]
    offs = [(0, 3), (3, 9), (9, 12), (12, 17), (17, 20)]
    assert name_token_indices(offs, 4, 17) == [1, 2, 3]
    with pytest.raises(ValueError):
        name_token_indices([(0, 0), (0, 3)], 10, 12)


@pytest.mark.slow
@pytest.mark.gpu
def test_qwen_extraction_tokens_and_shapes():
    """Real Qwen3.5-2B: name tokens are found inside the chat template; shapes match."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from phylogeo.extract import extract_names, load_model

    tok, model = load_model("Qwen/Qwen3.5-2B", revision="15852e8c16360a2fea060d615a32b45270f8a8fc")
    last, mean = extract_names(tok, model, ["house sparrow", "lion"], TEMPLATES[:2], chat=True)
    assert last.shape == (2, 2, 25, 2048)
    assert np.isfinite(last).all() and not np.allclose(last[0], last[1])
    # Layer 0 (embeddings) at the last name token is template-independent.
    assert np.allclose(last[0, 0, 0], last[0, 1, 0], atol=1e-3)
