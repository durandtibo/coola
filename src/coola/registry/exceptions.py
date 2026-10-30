r"""Define the exceptions raised by the registries."""

from __future__ import annotations

__all__ = ["AlreadyRegisteredError", "KeyNotRegisteredError", "TypeNotRegisteredError"]


class KeyNotRegisteredError(KeyError):
    r"""Raised when no value is registered for a key.

    It subclasses ``KeyError`` (a ``LookupError``). Registries whose
    keys are types raise the more specific ``TypeNotRegisteredError``.

    Example:
        ```pycon
        >>> from coola.registry import KeyNotRegisteredError, Registry
        >>> registry = Registry[str, int]()
        >>> try:
        ...     registry["missing"]
        ... except KeyNotRegisteredError:
        ...     print("not registered")
        ...
        not registered

        ```
    """

    def __str__(self) -> str:
        r"""Return the message without the quotes of ``KeyError``.

        Returns:
            The message of the error.
        """
        # KeyError.__str__ would repr() the message, escaping newlines.
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


class TypeNotRegisteredError(KeyNotRegisteredError):
    r"""Raised when no value is registered for a type or any of its
    parent types.

    It subclasses ``KeyNotRegisteredError`` (and so ``KeyError``), and it lets
    callers distinguish a missing type handler from an ordinary dict miss.

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


class AlreadyRegisteredError(ValueError):
    r"""Raised when registering a key that is already registered and
    ``exist_ok`` is ``False``.

    It is a ``ValueError`` (the value passed is not acceptable in the
    current state of the registry).

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
