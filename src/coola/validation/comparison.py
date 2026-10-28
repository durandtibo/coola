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

from typing import TYPE_CHECKING, Any

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
    _compare(lambda v: v < high, value, repr(high), name, f"less than {high!r}")


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
    _compare(lambda v: v >= 0, value, "0", name, "non-negative")


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
    _compare(lambda v: v > 0, value, "0", name, "positive")


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
    _compare(lambda v: v < 0, value, "0", name, "negative")


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
    _compare(lambda v: v <= 0, value, "0", name, "non-positive")


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
    _compare(
        lambda v: low < v < high,
        value,
        f"{low!r} and {high!r}",
        name,
        f"in the open range ({low!r}, {high!r})",
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
    _compare(
        lambda v: low <= v <= high,
        value,
        f"{low!r} and {high!r}",
        name,
        f"in the closed range [{low!r}, {high!r}]",
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
    _compare(
        lambda v: low <= v < high,
        value,
        f"{low!r} and {high!r}",
        name,
        f"in the half-open range [{low!r}, {high!r})",
    )
