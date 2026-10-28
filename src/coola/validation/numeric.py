r"""Contain functions to validate numeric values."""

from __future__ import annotations

__all__ = ["validate_finite", "validate_number"]

import math
import numbers
from typing import Any

from coola.validation.type import _type_error


def validate_finite(value: float, *, name: str = "value") -> None:
    r"""Validate that ``value`` is finite (neither infinite nor NaN).

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a real number (``bool`` is rejected).
        ValueError: If ``value`` is infinite or NaN.

    Example:
        ```pycon
        >>> from coola.validation import validate_finite
        >>> validate_finite(1.0)

        ```
    """
    # An int is always finite, but ``math.isfinite`` overflows for the huge ones.
    validate_number(value, name=name)
    if not isinstance(value, int) and not math.isfinite(value):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)


def validate_number(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a real number.

    Booleans are rejected, even though ``bool`` subclasses ``int``.
    The NumPy scalars (``numpy.float32``, ``numpy.int64``...) are
    accepted.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a real number or is a ``bool``.

    Example:
        ```pycon
        >>> from coola.validation import validate_number
        >>> validate_number(1)
        >>> validate_number(1.5)

        ```
    """
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        _type_error(value, "a real number", name)
