"""Prompt templates and location of the species-name tokens inside a prompt.

Global context: each species is embedded in 8 preregistered templates
(reports/preregistration.md section 2.5). For the chat model (Qwen3.5-2B) each
template is the content of one user turn of the official chat template; for
base models (Pythia) it is raw text with BOS prepended. The residual stream is
read at the last token of the species name, which is found from the tokenizer's
character offsets (so templates may continue after the name).
"""

from __future__ import annotations

from dataclasses import dataclass

# Preregistered templates: one mention, article "the", no class words.
TEMPLATES = [
    "The {name} is an animal.",
    "Here is a fact about the {name}.",
    "Scientists studied the {name} in the wild.",
    "This animal is called the {name}.",
    "Tell me about the {name}.",
    "A photo of the {name}.",
    "My favorite animal is the {name}.",
    "We read a book about the {name}.",
]

# Stage-4 ablation: the same 8 sentences for scientific binomials (English does not
# put "the" before a binomial, e.g. "Here is a fact about Panthera leo.").
SCI_TEMPLATES = [t.replace("the {name}", "{name}").replace("The {name}", "{name}") for t in TEMPLATES]

# Calibration concept sets (positive controls with known circular structure).
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
# Templates for calibration concepts (same structure: one mention, name inside).
CALIB_TEMPLATES = [
    "Today is {name}.",
    "Here is a fact about {name}.",
    "We met on {name}.",
    "This day is called {name}.",
    "Tell me about {name}.",
    "A photo taken on {name}.",
    "My favorite time is {name}.",
    "We read a book about {name}.",
]


@dataclass
class RenderedPrompt:
    """A fully formatted prompt string and the character span of the name in it."""

    text: str
    start: int  # char index of the first name character
    end: int  # char index one past the last name character


def render(template: str, name: str, tokenizer=None, chat: bool = False) -> RenderedPrompt:
    """Fill ``template`` with ``name`` and (optionally) wrap it in the chat template.

    Local: the name's character span is located inside the final string by
    finding the filled template inside the chat-formatted string, so it is exact
    even when the chat template adds prefixes.
    """
    content = template.format(name=name)
    name_start = template.index("{name}")
    if chat:
        text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=False)
        offset = text.index(content)
    else:
        text, offset = content, 0
    s = offset + name_start
    return RenderedPrompt(text=text, start=s, end=s + len(name))


def name_token_indices(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Indices of tokens whose character span overlaps [start, end).

    Local: tokens like "Ġsp" include a leading space before the name; overlap
    (not containment) is used so such tokens count as name tokens. Special
    tokens have (0, 0) offsets and never overlap a non-empty span.
    """
    idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not idx:
        raise ValueError(f"no token overlaps span [{start}, {end})")
    return idx
