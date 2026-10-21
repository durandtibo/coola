r"""Contain functions to validate types."""

from __future__ import annotations

__all__ = [
    "validate_bool",
    "validate_dict",
    "validate_float",
    "validate_frozenset",
    "validate_int",
    "validate_isinstance",
    "validate_list",
    "validate_set",
    "validate_str",
    "validate_tuple",
]

from typing import Any


def validate_bool(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a ``bool``.

    Integers such as ``0`` and ``1`` are rejected.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a ``bool``.

    Example:
        ```pycon
        >>> from coola.validation import validate_bool
        >>> validate_bool(True)

        ```
    """
    if not isinstance(value, bool):
        msg = f"{name} must be a bool, got {type(value)}"
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
    if not (isinstance(value, int) and not isinstance(value, bool)):
        msg = f"{name} must be an int, got {type(value)}"
        raise TypeError(msg)


def validate_float(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a float.

    Integers are rejected.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a float.

    Example:
        ```pycon
        >>> from coola.validation import validate_float
        >>> validate_float(1.0)

        ```
    """
    if not (isinstance(value, float)):
        msg = f"{name} must be a float, got {type(value)}"
        raise TypeError(msg)


def validate_str(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a str.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a str.

    Example:
        ```pycon
        >>> from coola.validation import validate_str
        >>> validate_str("abc")

        ```
    """
    if not (isinstance(value, str)):
        msg = f"{name} must be a str, got {type(value)}"
        raise TypeError(msg)


def validate_list(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a list.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a list.

    Example:
        ```pycon
        >>> from coola.validation import validate_list
        >>> validate_list([1, 2])

        ```
    """
    if not (isinstance(value, list)):
        msg = f"{name} must be a list, got {type(value)}"
        raise TypeError(msg)


def validate_tuple(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a tuple.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a tuple.

    Example:
        ```pycon
        >>> from coola.validation import validate_tuple
        >>> validate_tuple((1, 2))

        ```
    """
    if not (isinstance(value, tuple)):
        msg = f"{name} must be a tuple, got {type(value)}"
        raise TypeError(msg)


def validate_dict(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a dict.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a dict.

    Example:
        ```pycon
        >>> from coola.validation import validate_dict
        >>> validate_dict({"a": 1})

        ```
    """
    if not (isinstance(value, dict)):
        msg = f"{name} must be a dict, got {type(value)}"
        raise TypeError(msg)


def validate_set(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a set.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a set.

    Example:
        ```pycon
        >>> from coola.validation import validate_set
        >>> validate_set({1, 2})

        ```
    """
    if not (isinstance(value, set)):
        msg = f"{name} must be a set, got {type(value)}"
        raise TypeError(msg)


def validate_frozenset(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a frozenset.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a frozenset.

    Example:
        ```pycon
        >>> from coola.validation import validate_frozenset
        >>> validate_frozenset(frozenset({1, 2}))

        ```
    """
    if not (isinstance(value, frozenset)):
        msg = f"{name} must be a frozenset, got {type(value)}"
        raise TypeError(msg)


def validate_isinstance(value: Any, cls: type | tuple[type, ...], *, name: str = "value") -> None:
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
        msg = f"{name} must be an instance of {cls}, got {type(value)}"
        raise TypeError(msg)
