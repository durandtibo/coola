r"""Define the exceptions raised by the registries."""

from __future__ import annotations

__all__ = ["AlreadyRegisteredError", "TypeNotRegisteredError"]


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


class AlreadyRegisteredError(ValueError, RuntimeError):
    r"""Raised when registering a key that is already registered and
    ``exist_ok`` is ``False``.

    It is a ``ValueError`` (the value passed is not acceptable in the
    current state of the registry). It also derives from
    ``RuntimeError``, which this error used to be, so existing
    ``except RuntimeError`` code keeps working.

    Example:
        ```pycon
        >>> from coola.registry import AlreadyRegisteredError, Registry
        >>> registry = Registry[str, int]()
        >>> registry.register("key", 1)
        >>> try:
        ...     registry.register("key", 2)
        ... except AlreadyRegisteredError:
        ...     print("already registered")
        ...
        already registered

        ```
    """
