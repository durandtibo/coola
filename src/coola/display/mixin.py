r"""Provide display mixins for ``__repr__`` and ``__str__``
formatting."""

from __future__ import annotations

__all__ = ["InlineDisplayMixin", "MultilineDisplayMixin"]

from abc import ABC, abstractmethod
from typing import Any

from coola.utils.format import (
    repr_indent,
    repr_mapping,
    repr_mapping_line,
    str_indent,
    str_mapping,
    str_mapping_line,
)


class BaseDisplayMixin(ABC):
    """Abstract base for display mixins.

    Subclasses must implement :meth:`_get_repr_kwargs` to return a dict
    of constructor arguments used by ``__repr__`` and ``__str__``.
    """

    @abstractmethod
    def _get_repr_kwargs(self) -> dict[str, Any]:
        """Return a display-friendly dict of constructor arguments.

        Returns:
            A dict mapping argument names to their values, used to
            build the ``__repr__`` and ``__str__`` output.
        """


class MultilineDisplayMixin(BaseDisplayMixin):
    r"""Mixin that renders ``__repr__`` and ``__str__`` in multiline dict
    format.

    Each constructor argument is displayed on its own indented line.
    Best suited for objects with several arguments or deeply nested
    values where a single-line format would be hard to read.

    Example:
        ```pycon
        >>> from coola.display import MultilineDisplayMixin
        >>> from typing import Any
        >>> class MyClass(MultilineDisplayMixin):
        ...     def __init__(self, key1: str, key2: str) -> None:
        ...         self.key1 = key1
        ...         self.key2 = key2
        ...     def _get_repr_kwargs(self) -> dict[str, Any]:
        ...         return {"key1": self.key1, "key2": self.key2}
        ...
        >>> obj = MyClass(key1="value1", key2="value2")
        >>> repr(obj)
        'MyClass(\\n  (key1): value1\\n  (key2): value2\\n)'
        >>> str(obj)
        'MyClass(\\n  (key1): value1\\n  (key2): value2\\n)'

        ```
    """

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_repr_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_repr_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class InlineDisplayMixin(BaseDisplayMixin):
    """Mixin that renders ``__repr__`` and ``__str__`` on a single line.

    All constructor arguments are displayed inline. Best suited for
    simple objects with few, short arguments where a compact single-line
    format aids readability.

    Example:
        ```pycon
        >>> from coola.display import InlineDisplayMixin
        >>> from typing import Any
        >>> class MyClass(InlineDisplayMixin):
        ...     def __init__(self, key1: str, key2: str) -> None:
        ...         self.key1 = key1
        ...         self.key2 = key2
        ...     def _get_repr_kwargs(self) -> dict[str, Any]:
        ...         return {"key1": self.key1, "key2": self.key2}
        ...
        >>> obj = MyClass(key1="value1", key2="value2")
        >>> repr(obj)
        "MyClass(key1='value1', key2='value2')"
        >>> str(obj)
        'MyClass(key1=value1, key2=value2)'

        ```
    """

    def __repr__(self) -> str:
        args = repr_mapping_line(self._get_repr_kwargs())
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_mapping_line(self._get_repr_kwargs())
        return f"{self.__class__.__qualname__}({args})"
