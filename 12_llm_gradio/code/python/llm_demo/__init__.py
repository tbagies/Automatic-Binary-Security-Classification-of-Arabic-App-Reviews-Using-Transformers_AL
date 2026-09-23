"""CAMeLBERT semantic 1-NN demonstration package."""

from .core import (
    CameLBERTEncoder,
    classify_embeddings,
    load_datasets,
    select_examples,
    select_examples_by_semantic_theme,
)

__all__ = [
    "CameLBERTEncoder",
    "classify_embeddings",
    "load_datasets",
    "select_examples",
    "select_examples_by_semantic_theme",
]
