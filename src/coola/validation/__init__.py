r"""Contain functions to validate function/method arguments.

Every validator follows the same rules:

- It returns ``None`` when the value is valid, and never converts the value.
- A value of the wrong type raises ``TypeError``, and a value of the right type
    that breaks the constraint raises ``ValueError``.
- The message names the argument (``name``, a keyword-only option), the
    constraint and the value received.
- The type checks are strict (``bool`` is rejected by ``validate_int`` and
    ``validate_number``), the comparisons accept whatever supports them.
- A ``NaN`` is never valid.
- The package imports the standard library only.
"""

from __future__ import annotations

__all__ = [
    "validate_bool",
    "validate_dict",
    "validate_finite",
    "validate_float",
    "validate_frozenset",
    "validate_ge",
    "validate_gt",
    "validate_in",
    "validate_int",
    "validate_isinstance",
    "validate_le",
    "validate_list",
    "validate_lt",
    "validate_mapping",
    "validate_negative",
    "validate_non_negative",
    "validate_non_positive",
    "validate_not_empty",
    "validate_number",
    "validate_positive",
    "validate_range",
    "validate_set",
    "validate_str",
    "validate_tuple",
]

from coola.validation.comparison import (
    validate_ge,
    validate_gt,
    validate_le,
    validate_lt,
    validate_negative,
    validate_non_negative,
    validate_non_positive,
    validate_positive,
    validate_range,
)
from coola.validation.container import validate_in, validate_not_empty
from coola.validation.numeric import validate_finite, validate_number
from coola.validation.type import (
    validate_bool,
    validate_dict,
    validate_float,
    validate_frozenset,
    validate_int,
    validate_isinstance,
    validate_list,
    validate_mapping,
    validate_set,
    validate_str,
    validate_tuple,
)
