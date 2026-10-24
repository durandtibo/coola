r"""Contain functions to validate types.

The checks are strict and do not convert the value.
``validate_isinstance`` is the general function, and the other functions
are shortcuts for the most common builtin types, with the same error
message, so a developer does not have to write
``validate_isinstance(value, int)`` and remember the ``bool`` special
case.
"""

from __future__ import annotations

__all__ = [
    "validate_bool",
    "validate_dict",
    "validate_float",
    "validate_frozenset",
    "validate_int",
    "validate_isinstance",
    "validate_list",
    "validate_mapping",
    "validate_set",
    "validate_str",
    "validate_tuple",
]

from collections.abc import Mapping
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
    validate_isinstance(value, bool, name=name)


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
    validate_isinstance(value, float, name=name)


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
    validate_isinstance(value, str, name=name)


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
    validate_isinstance(value, list, name=name)


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
    validate_isinstance(value, tuple, name=name)


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
    validate_isinstance(value, dict, name=name)


def validate_mapping(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is a mapping.

    A mapping is an instance of ``collections.abc.Mapping``, for example
    a ``dict``, an ``OrderedDict`` or a ``MappingProxyType``.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a mapping.

    Example:
        ```pycon
        >>> from coola.validation import validate_mapping
        >>> validate_mapping({"a": 1})

        ```
    """
    validate_isinstance(value, Mapping, name=name)


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
    validate_isinstance(value, set, name=name)


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
    validate_isinstance(value, frozenset, name=name)


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
