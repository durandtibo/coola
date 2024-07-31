r"""Contain simple operations on nested data structure."""

from __future__ import annotations

__all__ = [
    "convert_to_dict_of_lists",
    "convert_to_list_of_dicts",
    "get_first_value",
    "remove_keys_starting_with",
    "to_flat_dict",
]

from coola.nested.conversion import convert_to_dict_of_lists, convert_to_list_of_dicts
from coola.nested.mapping import (
    get_first_value,
    remove_keys_starting_with,
    to_flat_dict,
)
