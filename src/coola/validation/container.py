r"""Contain functions to validate containers."""

from __future__ import annotations

__all__ = ["validate_in", "validate_not_empty"]

import reprlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Container, Sized

# The listing of the valid values is limited to 20 items, so the message stays short.
_LISTING = reprlib.Repr()
_LISTING.maxtuple = _LISTING.maxlist = _LISTING.maxset = _LISTING.maxfrozenset = 20
_LISTING.maxdict = _LISTING.maxdeque = 20


def validate_not_empty(value: Sized, *, name: str = "value") -> None:
    r"""Validate that ``value`` is not empty.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` has no length (for example an iterator,
            which is never consumed).
        ValueError: If ``value`` is empty.

    Example:
        ```pycon
        >>> from coola.validation import validate_not_empty
        >>> validate_not_empty([1, 2, 3])

        ```
    """
    try:
        is_empty = len(value) == 0
    except TypeError as exc:
        msg = f"{name} must have a length, got {type(value).__qualname__}"
        raise TypeError(msg) from exc
    if is_empty:
        msg = f"{name} must not be empty"
        raise ValueError(msg)


def validate_in(value: Any, valid: Container, *, name: str = "value") -> None:
    r"""Validate that ``value`` belongs to ``valid``.

    Args:
        value: The value to validate.
        valid: The container of valid values.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is not in ``valid`` (an unhashable
            value is not in a ``set``). The listing of ``valid`` in the
            message is truncated to 20 items. Pass a ``tuple`` for a
            stable listing, the order of a ``set`` is not stable.

    Example:
        ```pycon
        >>> from coola.validation import validate_in
        >>> validate_in("a", ("a", "b", "c"))

        ```
    """
    try:
        is_valid = value in valid
    except TypeError:
        if not (hasattr(valid, "__contains__") or hasattr(valid, "__iter__")):
            raise  # ``valid`` is not a container: a bug of the caller
        is_valid = False  # for example an unhashable value tested against a set
    if not is_valid:
        msg = f"{name} must be one of {_LISTING.repr(valid)}, got {value!r}"
        raise ValueError(msg)
