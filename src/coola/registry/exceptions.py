r"""Define the exceptions raised by the registries."""

from __future__ import annotations

__all__ = ["TypeNotRegisteredError"]


class TypeNotRegisteredError(KeyError, LookupError):
    r"""Raised when no value is registered for a type or any of its
    parent types.

    It subclasses ``KeyError`` so existing ``except KeyError`` code keeps
    working, but it lets callers distinguish a missing type handler from
    an ordinary dict miss.

    Example:
        ```pycon
        >>> from coola.registry import TypeNotRegisteredError, TypeRegistry
        >>> registry = TypeRegistry[str]()
        >>> try:
        ...     registry.resolve(int)
        ... except TypeNotRegisteredError:
        ...     print("not registered")
        ...
        not registered

        ```
    """

    def __str__(self) -> str:
        # KeyError.__str__ would repr() the message, escaping newlines.
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()
