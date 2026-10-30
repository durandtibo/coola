r"""Helpers to reshape, filter and convert nested data structures.

``nested`` is not one algorithm: it is a set of helpers built on the
traversal packages, in four groups. ``flat``: ``to_flat_dict`` and
``from_flat_dict``. ``keys``: ``remove_keys_if``,
``remove_keys_containing``, ``remove_keys_starting_with`` and
``add_prefix_suffix_to_keys``. ``mappings``: ``merge_mappings``,
``flatten_mapping`` and ``DuplicateKeyError``. ``records``:
``convert_to_dict_of_lists`` and ``convert_to_list_of_dicts``. The
conversion to JSON data is in ``coola.jsonable``.

Properties:

- Pure: the input is never modified. A function that rebuilds
    containers returns new ones; the leaves are shared, never copied.
- One notion of container: a function that looks inside the values
    enters the containers of the other packages (sequences, mappings,
    sets, dataclasses) and treats text, bytes, iterators and unknown
    types as leaves. ``to_flat_dict`` does not enter sets.
- Structure-preserving: a function that rebuilds a structure keeps the
    type of every container and the order of the entries.
- Any depth: no ``RecursionError``, the depth is bounded by memory only.
- Cycle-safe: a container that contains itself raises a ``ValueError``.
- Keys are labels: only the ``flat`` and ``keys`` functions read or
    change keys.
- No silent loss: two entries are never merged into one without a
    decision of the caller, a key collision raises. A wrong input type
    raises a ``TypeError`` that names the type.
- Documented round trips: ``from_flat_dict(to_flat_dict(x)) == x`` and
    ``convert_to_list_of_dicts(convert_to_dict_of_lists(r)) == r``
    under the conditions of the guide.
- Explicit options: options are keyword-only, an invalid option raises
    a ``ValueError`` before any work is done.
- Lazy: ``import coola.nested`` imports no optional library.
- Thread-safe: the state of a call is local to the call.
- Fast: the cost is linear in the number of nodes.

Not guaranteed: the cost of a container shared by several paths (it is
visited once per path), and the round trip of dataclasses and of
non-``str`` keys.

Several of these properties are reached step by step; the guide says
which ones.

Example:
    ```pycon
    >>> from coola.nested import to_flat_dict
    >>> to_flat_dict({"a": {"b": 1}, "c": [2, 3]})
    {'a.b': 1, 'c.0': 2, 'c.1': 3}

    ```
"""

from __future__ import annotations

__all__ = [
    "DuplicateKeyError",
    "add_prefix_suffix_to_keys",
    "convert_to_dict_of_lists",
    "convert_to_list_of_dicts",
    "flatten_frame",
    "flatten_mapping",
    "from_flat_dict",
    "merge_mappings",
    "remove_keys_containing",
    "remove_keys_if",
    "remove_keys_starting_with",
    "to_flat_dict",
]

from typing import TYPE_CHECKING

from coola.nested.flat import from_flat_dict, to_flat_dict
from coola.nested.keys import (
    add_prefix_suffix_to_keys,
    remove_keys_containing,
    remove_keys_if,
    remove_keys_starting_with,
)
from coola.nested.mapping import (
    DuplicateKeyError,
    flatten_mapping,
    merge_mappings,
)
from coola.nested.records import convert_to_dict_of_lists, convert_to_list_of_dicts
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.nested.polars import flatten_frame

# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {"flatten_frame": "coola.nested.polars"}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
