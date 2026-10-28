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

A validator exists only if it states a constraint that ``isinstance`` or one
comparison cannot state without a special case: ``validate_int`` exists for
the ``bool`` case, ``validate_number`` for the ``numbers.Real`` plus ``bool``
case, and ``validate_finite`` for the overflow of ``math.isfinite`` and the
NaN. ``validate_str(x)`` is ``validate_isinstance(x, str)`` and
``validate_positive(x)`` is ``validate_gt(x, 0)``. Validate an argument with
this package instead of writing ``raise ValueError`` for a type, range or
membership check.
"""

from __future__ import annotations

__all__ = [
    "validate_finite",
    "validate_ge",
    "validate_gt",
    "validate_in",
    "validate_int",
    "validate_isinstance",
    "validate_le",
    "validate_lt",
    "validate_not_empty",
    "validate_number",
    "validate_range",
]

from coola.validation.comparison import (
    validate_finite,
    validate_ge,
    validate_gt,
    validate_le,
    validate_lt,
    validate_range,
)
from coola.validation.container import validate_in, validate_not_empty
from coola.validation.type import validate_int, validate_isinstance, validate_number
