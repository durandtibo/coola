r"""Provide display mixins for ``__repr__`` and ``__str__``
formatting."""

from __future__ import annotations

__all__ = ["BaseDisplayMixin", "InlineDisplayMixin", "MultilineDisplayMixin"]

from abc import ABC, abstractmethod
from typing import Any

from coola.formatting.structure import (
    repr_indent,
    repr_mapping,
    repr_mapping_line,
    str_indent,
    str_mapping,
    str_mapping_line,
)


class BaseDisplayMixin(ABC):
    """Abstract base for display mixins.

    ``__repr__`` and ``__str__`` are defined here: they get the fields
    from ``_display_fields`` and delegate the layout to
    ``_render_fields``. Subclasses must implement ``_render_fields`` to
    choose the layout. Classes that use a layout mixin override
    ``_display_fields`` to return the values to show; by default no
    field is shown.
    """

    def __repr__(self) -> str:
        return self._render_fields(self._display_fields(), use_repr=True)

    def __str__(self) -> str:
        return self._render_fields(self._display_fields(), use_repr=False)

    def _display_fields(self) -> dict[str, Any]:
        """Return the fields to display.

        Returns:
            A dict mapping field names to their values, used to
                build the ``__repr__`` and ``__str__`` output.
                The values are formatted with ``repr`` in
                ``__repr__`` and with ``str`` in ``__str__``.
                By default, the dict is empty.
        """
        return {}

    @abstractmethod
    def _render_fields(self, fields: dict[str, Any], *, use_repr: bool) -> str:
        """Format the fields with the class name.

        Args:
            fields: The fields to display.
            use_repr: If ``True``, format the values with ``repr``,
                otherwise with ``str``.

        Returns:
            The formatted string.
        """


class MultilineDisplayMixin(BaseDisplayMixin):
    r"""Mixin that renders ``__repr__`` and ``__str__`` in multiline dict
    format.

    Each field is displayed on its own indented line. Best suited for
    objects with several fields or deeply nested values where a
    single-line format would be hard to read. Override
    ``_display_fields`` to choose the fields to show; by default none
    is shown.

    Example:
        ```pycon
        >>> from coola.display import MultilineDisplayMixin
        >>> from typing import Any
        >>> class MyClass(MultilineDisplayMixin):
        ...     def __init__(self, key1: str, key2: str) -> None:
        ...         self.key1 = key1
        ...         self.key2 = key2
        ...     def _display_fields(self) -> dict[str, Any]:
        ...         return {"key1": self.key1, "key2": self.key2}
        ...
        >>> obj = MyClass(key1="value1", key2="value2")
        >>> print(repr(obj))
        MyClass(
          (key1): value1
          (key2): value2
        )
        >>> print(str(obj))
        MyClass(
          (key1): value1
          (key2): value2
        )

        ```
    """

    def _render_fields(self, fields: dict[str, Any], *, use_repr: bool) -> str:
        args = repr_indent(repr_mapping(fields)) if use_repr else str_indent(str_mapping(fields))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class InlineDisplayMixin(BaseDisplayMixin):
    r"""Mixin that renders ``__repr__`` and ``__str__`` on a single line.

    All fields are displayed inline. Best suited for simple objects
    with few, short fields where a compact single-line format aids
    readability. Override ``_display_fields`` to choose the fields to
    show; by default none is shown, which suits stateless classes.

    Example:
        ```pycon
        >>> from coola.display import InlineDisplayMixin
        >>> from typing import Any
        >>> class MyClass(InlineDisplayMixin):
        ...     def __init__(self, key1: str, key2: str) -> None:
        ...         self.key1 = key1
        ...         self.key2 = key2
        ...     def _display_fields(self) -> dict[str, Any]:
        ...         return {"key1": self.key1, "key2": self.key2}
        ...
        >>> obj = MyClass(key1="value1", key2="value2")
        >>> print(repr(obj))
        MyClass(key1='value1', key2='value2')
        >>> print(str(obj))
        MyClass(key1=value1, key2=value2)
        >>> class NoFields(InlineDisplayMixin):
        ...     pass
        ...
        >>> print(repr(NoFields()))
        NoFields()

        ```
    """

    def _render_fields(self, fields: dict[str, Any], *, use_repr: bool) -> str:
        args = repr_mapping_line(fields) if use_repr else str_mapping_line(fields)
        return f"{self.__class__.__qualname__}({args})"
