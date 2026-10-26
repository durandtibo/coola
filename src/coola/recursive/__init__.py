r"""Utilities to recursively transform nested data structures.

Properties:

- Leaves only: ``func`` is called on the leaves, once per occurrence,
    never on a container.
- Structure-preserving: a container is rebuilt from the transformed
    children with its original type. Keys are never transformed.
- Type dispatch: the transformer is chosen from the type of the data,
    along the Method Resolution Order.
- Registry-extensible: a new type is supported by registering a
    transformer.
- Deterministic order: the leaves are visited depth first, left to right.
- Pure: the input is never modified.
- Cycle-safe: a container that contains itself raises a ``ValueError``.
- Errors propagate: an exception raised by ``func`` propagates unchanged.

Not guaranteed: laziness (the output is built whole), the cost of a
container shared by several paths (it is transformed once per path),
the order of the calls inside a set, and the depth (the recursion limit
of Python applies).

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
    "ConditionalTransformer",
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
from coola.recursive.conditional import ConditionalTransformer
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
