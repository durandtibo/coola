r"""Implement some utility functions to compute string representations
of quantities (byte sizes, durations, ratios)."""

from __future__ import annotations

__all__ = [
    "find_best_byte_unit",
    "make_bar",
    "str_human_byte_size",
    "str_time_human",
]

import datetime
import math

from coola.validation import validate_ge, validate_gt

BYTE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
    "TB": 1024 * 1024 * 1024 * 1024,
}


def make_bar(value: float, length: int = 10) -> str:
    r"""Create a progress bar string for a given value between 0 and 1.

    Args:
        value: A float between 0 and 1 representing the metric value.
        length: The length of the progress bar. Must be a positive integer.

    Returns:
        A string representing the progress bar.

    Raises:
        ValueError: If ``value`` is not in [0, 1], or ``length`` is not positive.

    Example:
        ```pycon
        >>> from coola.formatting.quantity import make_bar
        >>> make_bar(0.6)
        '[██████░░░░]'
        >>> make_bar(0.1)
        '[█░░░░░░░░░]'
        >>> make_bar(1.0)
        '[██████████]'
        >>> make_bar(0.0)
        '[░░░░░░░░░░]'

        ```
    """
    if not 0.0 <= value <= 1.0:
        msg = f"value must be in [0, 1], but got {value}"
        raise ValueError(msg)
    validate_gt(length, 0, name="length")

    filled = round(value * length)
    return f"[{'█' * filled}{'░' * (length - filled)}]"


def str_time_human(seconds: float) -> str:
    r"""Return a number of seconds in an easier format to read
    ``hh:mm:ss``.

    If the number of seconds is bigger than 1 day, this representation
    also encodes the number of days.

    Args:
        seconds: The number of seconds.

    Returns:
        The number of seconds in a string format (``hh:mm:ss``).
            A negative duration is prefixed with ``-``.

    Raises:
        ValueError: if ``seconds`` is not finite or too large.

    Example:
        ```pycon
        >>> from coola.formatting.quantity import str_time_human
        >>> str_time_human(1.2)
        '0:00:01.200000'
        >>> str_time_human(61.2)
        '0:01:01.200000'
        >>> str_time_human(3661.2)
        '1:01:01.200000'
        >>> str_time_human(-5)
        '-0:00:05'

        ```
    """
    if not math.isfinite(seconds):
        msg = f"seconds must be finite, got {seconds}"
        raise ValueError(msg)
    sign = "-" if seconds < 0 else ""
    try:
        return sign + str(datetime.timedelta(seconds=abs(seconds)))
    except OverflowError as exc:
        msg = f"seconds is too large to be represented, got {seconds}"
        raise ValueError(msg) from exc


def str_human_byte_size(size: int, unit: str | None = None) -> str:
    r"""Get a human-readable representation of the byte size.

    Args:
        size: The size in bytes.
        unit: The unit to use to show the byte size. If ``None``, the
            best unit is found automatically. The supported units
            are: ``'B'``, ``'KB'``, ``'MB'``, ``'GB'``, ``'TB'``.

    Returns:
        The byte size in a human-readable format.

    Raises:
        ValueError: if size is negative or unit is not supported.

    Example:
        ```pycon
        >>> from coola.formatting.quantity import str_human_byte_size
        >>> str_human_byte_size(2)
        '2.00 B'
        >>> str_human_byte_size(2048)
        '2.00 KB'
        >>> str_human_byte_size(2097152)
        '2.00 MB'
        >>> str_human_byte_size(2048, unit="B")
        '2,048.00 B'

        ```
    """
    validate_ge(size, 0, name="Size")
    if unit is None:  # Find the best unit.
        unit = find_best_byte_unit(size)
        # Pick the unit after rounding: 1048575 B is 1,024.00 KB, i.e. 1.00 MB.
        units = list(BYTE_UNITS)
        index = units.index(unit)
        if index + 1 < len(units) and round(size / BYTE_UNITS[unit], 2) >= 1024:
            unit = units[index + 1]
    if unit not in BYTE_UNITS:
        msg = f"Incorrect unit '{unit}'. The available units are {list(BYTE_UNITS.keys())}"
        raise ValueError(msg)
    return f"{size / BYTE_UNITS.get(unit, 1):,.2f} {unit}"


def find_best_byte_unit(size: int) -> str:
    r"""Return the best byte unit given the byte size.

    Args:
        size: The size in bytes.

    Returns:
        The best unit. The supported units are: ``'B'``, ``'KB'``,
            ``'MB'``, ``'GB'``, ``'TB'``.

    Raises:
        ValueError: if size is negative.

    Example:
        ```pycon
        >>> from coola.formatting.quantity import find_best_byte_unit
        >>> find_best_byte_unit(2)
        'B'
        >>> find_best_byte_unit(2048)
        'KB'
        >>> find_best_byte_unit(2097152)
        'MB'

        ```
    """
    validate_ge(size, 0, name="Size")
    best_unit = "B"
    for unit, multiplier in BYTE_UNITS.items():
        if (size / multiplier) >= 1:
            best_unit = unit
    return best_unit
