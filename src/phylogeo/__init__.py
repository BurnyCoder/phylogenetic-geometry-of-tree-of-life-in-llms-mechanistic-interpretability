"""phylogeo: phylogenetic geometry of animal-species representations in LLMs.

Global context: the package reads the hidden states an LLM computes for animal
species names and tests whether their geometry matches the tree of life
(Open Tree of Life topology). Entry point: ``python -m phylogeo.run``.
"""

__version__ = "0.1.0"

# Load .env (HF_HOME, DATA_DIR, ...) on any import of the package, before any
# submodule imports transformers, so models are always cached on the large drive.
from phylogeo import config as _config  # noqa: E402,F401
