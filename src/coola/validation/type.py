r"""Contain functions to validate types.

The checks are strict and do not convert the value.
``validate_isinstance`` is the general function. ``validate_int`` and
``validate_number`` exist because they reject ``bool``, which subclasses
``int``. A validator is added only if it states a constraint that
``isinstance`` or one comparison cannot state without a special case.
"""

from __future__ import annotations

__all__ = [
    "validate_int",
    "validate_isinstance",
    "validate_number",
]

import numbers
from typing import TYPE_CHECKING, Any, NoReturn, get_args

if TYPE_CHECKING:
    from types import UnionType


def _type_name(cls: Any) -> str:
    r"""Return the name of a type, a tuple of types or a union, for an
    error message.

    Args:
        cls: The type, tuple of types or union.

    Returns:
        The ``__qualname__`` of a class, or the names joined by ``or``.
    """
    if isinstance(cls, tuple):
        return " or ".join(_type_name(c) for c in cls)
    if args := get_args(cls):  # ``int | str``
        return " or ".join(_type_name(c) for c in args)
    return getattr(cls, "__qualname__", str(cls))


def _type_error(value: Any, expected: str, name: str) -> NoReturn:
    r"""Raise the ``TypeError`` of a validator.

    Args:
        value: The value that has the wrong type.
        expected: The expected type, with its article (for example
            ``an int``).
        name: The name of the value, used in the error message.

    Raises:
        TypeError: Always.
    """
    msg = f"{name} must be {expected}, got {type(value).__qualname__}"
    raise TypeError(msg)


def validate_int(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is an int.

    Booleans are rejected, even though ``bool`` subclasses ``int``.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not an int.

    Example:
        ```pycon
        >>> from coola.validation import validate_int
        >>> validate_int(1)

        ```
    """
    # ``bool`` is a subclass of ``int``, but ``True`` is rarely meant as the number 1
    # (for example for a size or a count), so it is rejected like in the other
    # strict checks of this module. Use ``validate_isinstance(value, int)`` to accept it.
    if not (isinstance(value, int) and not isinstance(value, bool)):
        _type_error(value, "an int", name)


def validate_isinstance(
    value: Any, cls: type | tuple[type, ...] | UnionType, *, name: str = "value"
) -> None:
    r"""Validate that ``value`` is an instance of ``cls``.

    Args:
        value: The value to validate.
        cls: The type or tuple of types that ``value`` must be an
            instance of.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not an instance of ``cls``.

    Example:
        ```pycon
        >>> from coola.validation import validate_isinstance
        >>> validate_isinstance(1, int)

        ```
    """
    if not isinstance(value, cls):
        _type_error(value, f"an instance of {_type_name(cls)}", name)


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
    # The exact ``int`` and ``float`` are the common cases: the ``numbers.Real`` ABC check is slow.
    if type(value) is int or type(value) is float:
        return
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        _type_error(value, "a real number", name)
