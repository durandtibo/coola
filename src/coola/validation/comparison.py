r"""Contain functions to validate comparisons."""

from __future__ import annotations

__all__ = [
    "validate_finite",
    "validate_ge",
    "validate_gt",
    "validate_le",
    "validate_lt",
    "validate_range",
]

import math
from typing import TYPE_CHECKING, Any

from coola.validation.type import validate_number

if TYPE_CHECKING:
    from collections.abc import Callable


def _compare(
    test: Callable[[Any], Any], value: Any, bounds: str, name: str, requirement: str
) -> None:
    r"""Evaluate a comparison and raise a clear error if it fails.

    Args:
        test: The function that evaluates the comparison of a value.
        value: The value to validate.
        bounds: The representation of the bounds, used in the error message.
        name: The name of the value, used in the error message.
        requirement: The requirement, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared with the bounds (for
            example ``None``, a ``str`` or an array with several items).
        ValueError: If the comparison is false (or ``value`` is NaN).
    """
    try:
        is_valid = bool(test(value))
    except (TypeError, ValueError) as exc:
        # ValueError: the truth value of an array with several items is ambiguous.
        msg = f"{name} cannot be compared with {bounds}, got {value!r}"
        raise TypeError(msg) from exc
    if not is_valid:
        msg = f"{name} must be {requirement}, got {value!r}"
        raise ValueError(msg)


def validate_ge(value: Any, low: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is greater than or equal to ``low``.

    Args:
        value: The value to validate.
        low: The lower bound, inclusive.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is lower than ``low``.

    Example:
        ```pycon
        >>> from coola.validation import validate_ge
        >>> validate_ge(1, 0)

        ```
    """
    # The success path is a plain comparison; ``_compare`` evaluates it again to build the error.
    try:
        if value >= low:
            return
    except (TypeError, ValueError):
        pass  # reported by ``_compare``
    _compare(lambda v: v >= low, value, repr(low), name, f"greater than or equal to {low!r}")


def validate_gt(value: Any, low: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly greater than ``low``.

    Args:
        value: The value to validate.
        low: The lower bound, exclusive.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is lower than or equal to ``low``.

    Example:
        ```pycon
        >>> from coola.validation import validate_gt
        >>> validate_gt(1, 0)

        ```
    """
    # The success path is a plain comparison; ``_compare`` evaluates it again to build the error.
    try:
        if value > low:
            return
    except (TypeError, ValueError):
        pass  # reported by ``_compare``
    _compare(lambda v: v > low, value, repr(low), name, f"greater than {low!r}")


def validate_le(value: Any, high: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is less than or equal to ``high``.

    Args:
        value: The value to validate.
        high: The upper bound, inclusive.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is greater than ``high``.

    Example:
        ```pycon
        >>> from coola.validation import validate_le
        >>> validate_le(0, 1)

        ```
    """
    # The success path is a plain comparison; ``_compare`` evaluates it again to build the error.
    try:
        if value <= high:
            return
    except (TypeError, ValueError):
        pass  # reported by ``_compare``
    _compare(lambda v: v <= high, value, repr(high), name, f"less than or equal to {high!r}")


def validate_lt(value: Any, high: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly less than ``high``.

    Args:
        value: The value to validate.
        high: The upper bound, exclusive.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is greater than or equal to ``high``.

    Example:
        ```pycon
        >>> from coola.validation import validate_lt
        >>> validate_lt(0, 1)

        ```
    """
    # The success path is a plain comparison; ``_compare`` evaluates it again to build the error.
    try:
        if value < high:
            return
    except (TypeError, ValueError):
        pass  # reported by ``_compare``
    _compare(lambda v: v < high, value, repr(high), name, f"less than {high!r}")


def validate_range(
    value: Any,
    low: Any,
    high: Any,
    *,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
    name: str = "value",
) -> None:
    r"""Validate that ``value`` is in the range from ``low`` to ``high``.

    By default, the range is closed (``low <= value <= high``). The two
    flags exclude a bound, which gives the open range ``low < value <
    high`` or a half-open range. The range is written with the usual
    notation in the error message, for example ``[0, 1)``.

    Args:
        value: The value to validate.
        low: The lower bound.
        high: The upper bound.
        low_inclusive: If ``True``, ``low`` is a valid value.
        high_inclusive: If ``True``, ``high`` is a valid value.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared with the bounds (for
            example ``None``, a ``str`` or an array with several items).
        ValueError: If the range is empty (for example ``low > high``),
            or if ``value`` is not in the range (or is NaN).

    Example:
        ```pycon
        >>> from coola.validation import validate_range
        >>> validate_range(0.5, 0, 1)
        >>> validate_range(0, 0, 1, high_inclusive=False)

        ```
    """
    # The success path is a plain comparison. A value in the range proves that the range is not
    # empty, so the bounds are checked only on the failure path.
    try:
        if (value >= low if low_inclusive else value > low) and (
            value <= high if high_inclusive else value < high
        ):
            return
    except (TypeError, ValueError):
        pass  # reported by ``_compare``
    left = "[" if low_inclusive else "("
    right = "]" if high_inclusive else ")"
    interval = f"{left}{low!r}, {high!r}{right}"
    # A range that no value can satisfy is a bug of the caller: report the bounds, not the value.
    try:
        is_empty = low > high or (low == high and not (low_inclusive and high_inclusive))
    except (TypeError, ValueError):
        is_empty = False  # the bounds are not comparable with each other
    if is_empty:
        msg = f"{name}: the range {interval} is empty"
        raise ValueError(msg)
    _compare(
        lambda v: (
            (v >= low if low_inclusive else v > low) and (v <= high if high_inclusive else v < high)
        ),
        value,
        f"{low!r} and {high!r}",
        name,
        f"in the range {interval}",
    )


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
    # Fast path for the common cases: a finite float, or an int (always finite, but
    # ``math.isfinite`` overflows for the huge ones).
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return
    validate_number(value, name=name)
    if not isinstance(value, int) and not math.isfinite(value):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)
