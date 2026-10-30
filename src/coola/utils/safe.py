r"""Conversions of objects to strings that never raise.

These helpers are meant for error messages, logs and reports, where a
broken ``__str__`` or ``__repr__`` (or an ``int`` above the string
conversion limit) must not hide the information being reported.
"""

from __future__ import annotations

__all__ = ["safe_repr", "safe_str"]


def safe_str(obj: object) -> str:
    r"""Return ``str(obj)``, or a placeholder if the conversion fails.

    Args:
        obj: The object to convert.

    Returns:
        The string representation of the object, or
            ``<ClassName object at 0x...>`` if ``str`` raised an
            exception.

    Example:
        ```pycon
        >>> from coola.utils.safe import safe_str
        >>> safe_str(42)
        '42'
        >>> class Bad:
        ...     def __str__(self):
        ...         raise RuntimeError
        ...
        >>> safe_str(Bad()).startswith("<Bad object at 0x")
        True

        ```
    """
    try:
        return str(obj)
    except Exception:  # noqa: BLE001
        return _placeholder(obj)


def safe_repr(obj: object) -> str:
    r"""Return ``repr(obj)``, or a placeholder if the conversion fails.

    Args:
        obj: The object to convert.

    Returns:
        The representation of the object, or
            ``<ClassName object at 0x...>`` if ``repr`` raised an
            exception.

    Example:
        ```pycon
        >>> from coola.utils.safe import safe_repr
        >>> safe_repr("abc")
        "'abc'"
        >>> class Bad:
        ...     def __repr__(self):
        ...         raise RuntimeError
        ...
        >>> safe_repr(Bad()).startswith("<Bad object at 0x")
        True

        ```
    """
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001
        return _placeholder(obj)


def _placeholder(obj: object) -> str:
    r"""Create a placeholder for an object that cannot be formatted.

    Args:
        obj: The object.

    Returns:
        The type and the address of the object, e.g. ``<Foo object at
            0x7f...>``.
    """
    return f"<{type(obj).__qualname__} object at {id(obj):#x}>"
