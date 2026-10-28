r"""Helpers to reshape and query nested mapping data structures."""

from __future__ import annotations

__all__ = [
    "DuplicateKeyError",
    "add_prefix_suffix_to_keys",
    "convert_to_dict_of_lists",
    "convert_to_jsonable",
    "convert_to_list_of_dicts",
    "flatten_frame",
    "flatten_mapping",
    "from_flat_dict",
    "get_first_value",
    "merge_mappings",
    "remove_keys_containing",
    "remove_keys_if",
    "remove_keys_starting_with",
    "to_flat_dict",
    "try_sorted_list",
]

from typing import TYPE_CHECKING

from coola.nested.conversion import (
    convert_to_dict_of_lists,
    convert_to_jsonable,
    convert_to_list_of_dicts,
    try_sorted_list,
)
from coola.nested.flat import from_flat_dict, to_flat_dict
from coola.nested.mapping import (
    DuplicateKeyError,
    add_prefix_suffix_to_keys,
    flatten_mapping,
    get_first_value,
    merge_mappings,
    remove_keys_containing,
    remove_keys_if,
    remove_keys_starting_with,
)
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.nested.polars import flatten_frame

# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {"flatten_frame": "coola.nested.polars"}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
