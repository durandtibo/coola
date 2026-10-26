r"""Utilities to recursively transform nested data structures.

Example:
    ```pycon
    >>> from coola.recursive import recursive_apply
    >>> recursive_apply({"a": 1, "b": 2}, lambda x: x * 10)
    {'a': 10, 'b': 20}

    ```
"""

from __future__ import annotations

__all__ = [
    "BaseTransformer",
    "DataclassTransformer",
    "DefaultTransformer",
    "IdentityTransformer",
    "KeyFilterTransformer",
    "MappingTransformer",
    "SequenceTransformer",
    "SetTransformer",
    "Transformer",
    "TransformerRegistry",
    "enter_dataclasses",
    "get_default_registry",
    "get_default_transformer_registry",
    "recursive_apply",
    "register_transformers",
]

from coola.recursive.base import BaseTransformer, Transformer
from coola.recursive.dataclass import DataclassTransformer
from coola.recursive.default import DefaultTransformer
from coola.recursive.identity import IdentityTransformer
from coola.recursive.interface import (
    enter_dataclasses,
    get_default_registry,
    get_default_transformer_registry,
    recursive_apply,
    register_transformers,
)
from coola.recursive.key import KeyFilterTransformer
from coola.recursive.mapping import MappingTransformer
from coola.recursive.registry import TransformerRegistry
from coola.recursive.sequence import SequenceTransformer
from coola.recursive.set import SetTransformer
