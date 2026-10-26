r"""Contain functions to validate comparisons."""

from __future__ import annotations

__all__ = [
    "validate_closed_range",
    "validate_ge",
    "validate_gt",
    "validate_half_open_range",
    "validate_le",
    "validate_lt",
    "validate_negative",
    "validate_non_negative",
    "validate_non_positive",
    "validate_open_range",
    "validate_positive",
]

import operator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _check(
    compare: Callable[[Any, Any], Any], value: Any, bound: Any, name: str, requirement: str
) -> None:
    r"""Check a comparison and raise a clear error if it fails.

    Args:
        compare: The comparison function (e.g. ``operator.ge``).
        value: The value to validate.
        bound: The bound to compare ``value`` with.
        name: The name of the value, used in the error message.
        requirement: The requirement, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared with ``bound`` (for
            example ``None``, a ``str`` or an array with several items).
        ValueError: If the comparison is false (or ``value`` is NaN).
    """
    try:
        is_valid = bool(compare(value, bound))
    except (TypeError, ValueError) as exc:
        # ValueError: the truth value of an array with several items is ambiguous.
        msg = f"{name} must be a number that can be compared with {bound}, got {value!r}"
        raise TypeError(msg) from exc
    if not is_valid:
        msg = f"{name} must be {requirement}, got {value}"
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
    _check(operator.ge, value, low, name, f"greater than or equal to {low}")


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
    _check(operator.gt, value, low, name, f"greater than {low}")


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
    _check(operator.le, value, high, name, f"less than or equal to {high}")


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
    _check(operator.lt, value, high, name, f"less than {high}")


def validate_non_negative(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is not negative.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is negative.

    Example:
        ```pycon
        >>> from coola.validation import validate_non_negative
        >>> validate_non_negative(1)

        ```
    """
    _check(operator.ge, value, 0, name, "non-negative")


def validate_positive(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly positive.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is not positive.

    Example:
        ```pycon
        >>> from coola.validation import validate_positive
        >>> validate_positive(1)

        ```
    """
    _check(operator.gt, value, 0, name, "positive")


def validate_negative(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly negative.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is not negative.

    Example:
        ```pycon
        >>> from coola.validation import validate_negative
        >>> validate_negative(-1)

        ```
    """
    _check(operator.lt, value, 0, name, "negative")


def validate_non_positive(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is not positive.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is positive.

    Example:
        ```pycon
        >>> from coola.validation import validate_non_positive
        >>> validate_non_positive(-1)

        ```
    """
    _check(operator.le, value, 0, name, "non-positive")


def _check_range(
    value: Any, low: Any, high: Any, name: str, requirement: str, *, low_open: bool, high_open: bool
) -> None:
    r"""Check that a value is in a range and raise a clear error if it is
    not.

    Args:
        value: The value to validate.
        low: The lower bound.
        high: The upper bound.
        name: The name of the value, used in the error message.
        requirement: The requirement, used in the error message.
        low_open: If ``True``, the lower bound is excluded.
        high_open: If ``True``, the upper bound is excluded.

    Raises:
        TypeError: If ``value`` cannot be compared with the bounds.
        ValueError: If ``value`` is not in the range (or is NaN).
    """
    lower = operator.gt if low_open else operator.ge
    upper = operator.lt if high_open else operator.le
    try:
        is_valid = bool(lower(value, low)) and bool(upper(value, high))
    except (TypeError, ValueError) as exc:
        # ValueError: the truth value of an array with several items is ambiguous.
        msg = f"{name} must be a number that can be compared with {low} and {high}, got {value!r}"
        raise TypeError(msg) from exc
    if not is_valid:
        msg = f"{name} must be {requirement}, got {value}"
        raise ValueError(msg)


def validate_open_range(value: Any, low: Any, high: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is in the open range ``low < value <
    high``.

    Args:
        value: The value to validate.
        low: The lower bound, exclusive.
        high: The upper bound, exclusive.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is not in the range (or is NaN).

    Example:
        ```pycon
        >>> from coola.validation import validate_open_range
        >>> validate_open_range(0.5, 0, 1)

        ```
    """
    _check_range(
        value,
        low,
        high,
        name,
        f"in the open range ({low}, {high})",
        low_open=True,
        high_open=True,
    )


def validate_closed_range(value: Any, low: Any, high: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is in the closed range ``low <= value <=
    high``.

    Args:
        value: The value to validate.
        low: The lower bound, inclusive.
        high: The upper bound, inclusive.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is not in the range (or is NaN).

    Example:
        ```pycon
        >>> from coola.validation import validate_closed_range
        >>> validate_closed_range(1, 0, 1)

        ```
    """
    _check_range(
        value,
        low,
        high,
        name,
        f"in the closed range [{low}, {high}]",
        low_open=False,
        high_open=False,
    )


def validate_half_open_range(value: Any, low: Any, high: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is in the half-open range ``low <= value
    < high``.

    Args:
        value: The value to validate.
        low: The lower bound, inclusive.
        high: The upper bound, exclusive.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` cannot be compared (for example ``None``, a
            ``str`` or an array with several items).
        ValueError: If ``value`` is not in the range (or is NaN).

    Example:
        ```pycon
        >>> from coola.validation import validate_half_open_range
        >>> validate_half_open_range(0, 0, 1)

        ```
    """
    _check_range(
        value,
        low,
        high,
        name,
        f"in the half-open range [{low}, {high})",
        low_open=False,
        high_open=True,
    )
