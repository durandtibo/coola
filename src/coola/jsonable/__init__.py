r"""Utilities to convert nested data to JSON data.

``to_jsonable`` turns any nested data into the values that
``json.loads`` returns, with one function, one registry and one
transformer per type.

Properties:

- JSON data model: the output contains only ``dict`` with ``str`` keys,
    ``list``, ``str``, ``int``, ``float``, ``bool`` and ``None``. A
    subclass of ``str``, ``int`` or ``float`` (an ``IntEnum`` member) is
    kept: ``json`` writes it as its base value.
- Accepted by ``json``: with ``on_unsupported="raise"``,
    ``json.dumps(to_jsonable(x))`` never raises, and the function is
    idempotent.
- Strict and loud: a value whose type has no transformer raises a
    ``TypeError`` that names the type. ``"warn"`` and ``"keep"`` are
    explicit opt-outs.
- Type dispatch: the transformer is chosen from the type of the data,
    along the Method Resolution Order.
- Registry-extensible: a new type is supported by registering a
    transformer; the plugin group is ``coola.jsonable.transformers``.
- Keys are strings: a key is converted by the registry, then written
    like ``json.dumps`` writes it. A key that cannot be a string raises
    a ``TypeError`` and two keys that give the same string raise a
    ``ValueError``.
- Deterministic: a set becomes a sorted list, dictionaries keep their
    order, so the output is the same in every process.
- Pure: the input is never modified and the output shares no container
    with it.
- Cycle-safe: a container that contains itself raises a ``ValueError``.
- Any depth: no ``RecursionError``, the traversal uses an explicit
    stack.
- Consistent with siblings: dataclasses are entered, text and bytes are
    leaves, options are keyword-only.

Not guaranteed: a round trip to the Python types, strict JSON
(``NaN`` and the infinities are kept), the sharing of a container
reachable by several paths (it is converted once per path).

Example:
    ```pycon
    >>> from coola.jsonable import to_jsonable
    >>> to_jsonable({"a": (1, 2), "b": {3, 1, 2}})
    {'a': [1, 2], 'b': [1, 2, 3]}

    ```
"""

from __future__ import annotations

__all__ = [
    "ConvertTransformer",
    "DataclassTransformer",
    "MappingTransformer",
    "SequenceTransformer",
    "SetTransformer",
    "get_default_registry",
    "register_transformers",
    "to_jsonable",
]

from coola.jsonable.interface import get_default_registry, register_transformers, to_jsonable
from coola.jsonable.transformers import (
    ConvertTransformer,
    DataclassTransformer,
    MappingTransformer,
    SequenceTransformer,
    SetTransformer,
)
