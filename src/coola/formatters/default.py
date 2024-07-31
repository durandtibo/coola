r"""Implement the default formatter for some of the native types."""

from __future__ import annotations

__all__ = [
    "DefaultFormatter",
    "MappingFormatter",
    "SequenceFormatter",
    "SetFormatter",
]

from collections.abc import Mapping, Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any, TypeVar

from coola.formatters.base import BaseFormatter
from coola.utils.format import str_indent, str_mapping, str_sequence

if TYPE_CHECKING:
    from coola.summarizers.base import BaseSummarizer

T = TypeVar("T")


class DefaultFormatter(BaseFormatter[Any]):
    r"""Implement the default formatter.

    Args:
        max_characters: The maximum number of characters to
            show. If a negative value is provided, all the characters
            are shown.

    Example usage:

    ```pycon

    >>> from coola import Summarizer
    >>> from coola.formatters import DefaultFormatter
    >>> formatter = DefaultFormatter()
    >>> formatter.format(Summarizer(), 1)
    <class 'int'> 1

    ```
    """

    def __init__(self, max_characters: int = -1) -> None:
        self.set_max_characters(max_characters)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(max_characters={self._max_characters:,})"

    def clone(self) -> DefaultFormatter:
        return self.__class__(max_characters=self._max_characters)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._max_characters == other._max_characters

    def format(
        self,
        summarizer: BaseSummarizer,  # noqa: ARG002
        value: Any,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        if depth >= max_depth:
            return self._format(str(value))
        return f"{type(value)} {self._format(str(value))}"

    def _format(self, value: str) -> str:
        if self._max_characters >= 0 and len(value) > self._max_characters:
            value = value[: self._max_characters] + "..."
        return value

    def load_state_dict(self, state: dict) -> None:
        self._max_characters = state["max_characters"]

    def state_dict(self) -> dict:
        return {"max_characters": self._max_characters}

    def get_max_characters(self) -> int:
        r"""Get the maximum number of characters to show.

        Returns:
            The maximum number of characters to show.

        Example usage:

        ```pycon
        >>> from coola.formatters import DefaultFormatter
        >>> formatter = DefaultFormatter()
        >>> formatter.get_max_characters()
        -1

        ```
        """
        return self._max_characters

    def set_max_characters(self, max_characters: int) -> None:
        r"""Set the maximum number of characters to show.

        Args:
            max_characters: The maximum number of
                characters to show.

        Raises:
            TypeError: if ``max_characters`` is not an integer.

        Example usage:

        ```pycon
        >>> from coola.formatters import DefaultFormatter
        >>> formatter = DefaultFormatter()
        >>> formatter.set_max_characters(10)
        >>> formatter.get_max_characters()
        10

        ```
        """
        if not isinstance(max_characters, int):
            msg = (
                "Incorrect type for max_characters. Expected int value but "
                f"received {max_characters}"
            )
            raise TypeError(msg)
        self._max_characters = max_characters


class BaseCollectionFormatter(BaseFormatter[T]):
    r"""Implement a base class to implement a formatter for
    ``Collection``.

    Args:
        max_items (int, optional): The maximum number
            of items to show. If a negative value is provided,
            all the items are shown.
        num_spaces (int, optional): The number of spaces
            used for the indentation.

    Example usage:

    ```pycon

    >>> from coola import Summarizer
    >>> from coola.formatters import MappingFormatter
    >>> formatter = MappingFormatter()
    >>> print(formatter.format(Summarizer(), {"key1": 1.2, "key2": "abc", "key3": 42}))
    <class 'dict'> (length=3)
      (key1): 1.2
      (key2): abc
      (key3): 42

    ```
    """

    def __init__(self, max_items: int = 5, num_spaces: int = 2) -> None:
        self.set_max_items(max_items)
        self.set_num_spaces(num_spaces)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(max_items={self._max_items:,}, "
            f"num_spaces={self._num_spaces})"
        )

    def clone(self) -> BaseCollectionFormatter:
        return self.__class__(max_items=self._max_items, num_spaces=self._num_spaces)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._max_items == other._max_items and self._num_spaces == other._num_spaces

    def load_state_dict(self, state: dict) -> None:
        self._max_items = state["max_items"]
        self._num_spaces = state["num_spaces"]

    def state_dict(self) -> dict:
        return {"max_items": self._max_items, "num_spaces": self._num_spaces}

    def get_max_items(self) -> int:
        r"""Get the maximum number of items to show.

        Returns:
            The maximum number of items to show.

        Example usage:

        ```pycon
        >>> from coola.formatters import MappingFormatter
        >>> formatter = MappingFormatter()
        >>> formatter.get_max_items()
        5

        ```
        """
        return self._max_items

    def set_max_items(self, max_items: int) -> None:
        r"""Set the maximum number of items to show.

        Args:
            max_items: The maximum number of items to show.

        Raises:
            TypeError: if ``max_items`` is not an integer.

        Example usage:

        ```pycon
        >>> from coola.formatters import MappingFormatter
        >>> formatter = MappingFormatter()
        >>> formatter.set_max_items(10)
        >>> formatter.get_max_items()
        10

        ```
        """
        if not isinstance(max_items, int):
            msg = f"Incorrect type for max_items. Expected int value but received {max_items}"
            raise TypeError(msg)
        self._max_items = max_items

    def get_num_spaces(self) -> int:
        r"""Get the number of spaces for indentation.

        Returns:
            The number of spaces for indentation.

        Example usage:

        ```pycon
        >>> from coola.formatters import MappingFormatter
        >>> formatter = MappingFormatter()
        >>> formatter.get_num_spaces()
        2

        ```
        """
        return self._num_spaces

    def set_num_spaces(self, num_spaces: int) -> None:
        r"""Set the number of spaces for indentation.

        Args:
            num_spaces: The number of spaces for indentation.

        Raises:
            TypeError: if ``num_spaces`` is not an integer.
            ValueError: if ``num_spaces`` is not a positive integer.

        Example usage:

        ```pycon
        >>> from coola.formatters import MappingFormatter
        >>> formatter = MappingFormatter()
        >>> formatter.set_num_spaces(4)
        >>> formatter.get_num_spaces()
        4

        ```
        """
        if not isinstance(num_spaces, int):
            msg = f"Incorrect type for num_spaces. Expected int value but received {num_spaces}"
            raise TypeError(msg)
        if num_spaces < 0:
            msg = (
                "Incorrect value for num_spaces. Expected a positive integer value but "
                f"received {num_spaces}"
            )
            raise ValueError(msg)
        self._num_spaces = num_spaces


class MappingFormatter(BaseCollectionFormatter[Mapping]):
    r"""Implement a formatter for ``Mapping``.

    Example usage:

    ```pycon

    >>> from coola import Summarizer
    >>> from coola.formatters import MappingFormatter
    >>> formatter = MappingFormatter()
    >>> print(formatter.format(Summarizer(), {"key1": 1.2, "key2": "abc", "key3": 42}))
    <class 'dict'> (length=3)
      (key1): 1.2
      (key2): abc
      (key3): 42

    ```
    """

    def format(
        self, summarizer: BaseSummarizer, value: Mapping, depth: int = 0, max_depth: int = 1
    ) -> str:
        if depth >= max_depth:
            return summarizer.summary(str(value), depth=depth + 1, max_depth=max_depth)
        typ = type(value)
        length = len(value)
        if length > 0:
            items = value.items()
            if self._max_items >= 0:
                items = islice(value.items(), self._max_items)
            value = str_mapping(
                {
                    key: summarizer.summary(val, depth=depth + 1, max_depth=max_depth)
                    for key, val in items
                },
                num_spaces=self._num_spaces,
            )
            if length > self._max_items and self._max_items >= 0:
                value = f"{value}\n..."
            value = f"(length={length:,})\n{value}"
        return str_indent(f"{typ} {value}", num_spaces=self._num_spaces)


class SequenceFormatter(BaseCollectionFormatter[Sequence]):
    r"""Implement a formatter for ``Sequence``.

    Example usage:

    ```pycon

    >>> from coola import Summarizer
    >>> from coola.formatters import SequenceFormatter
    >>> formatter = SequenceFormatter()
    >>> print(formatter.format(Summarizer(), [1, 2, 3]))
    <class 'list'> (length=3)
      (0): 1
      (1): 2
      (2): 3

    ```
    """

    def format(
        self, summarizer: BaseSummarizer, value: Sequence, depth: int = 0, max_depth: int = 1
    ) -> str:
        if depth >= max_depth:
            return summarizer.summary(str(value), depth=depth + 1, max_depth=max_depth)
        typ = type(value)
        length = len(value)
        if length > 0:
            if self._max_items >= 0:
                value = value[: self._max_items]
            value = str_sequence(
                [summarizer.summary(val, depth=depth + 1, max_depth=max_depth) for val in value],
                num_spaces=self._num_spaces,
            )
            if length > self._max_items and self._max_items > 0:
                value = f"{value}\n..."
            value = f"(length={length:,})\n{value}"
        return str_indent(f"{typ} {value}", num_spaces=self._num_spaces)


class SetFormatter(BaseCollectionFormatter[set]):
    r"""Implement a formatter for ``set``.

    Example usage:

    ```pycon

    >>> from coola import Summarizer
    >>> from coola.formatters import SetFormatter
    >>> formatter = SetFormatter()
    >>> formatter.format(Summarizer(), {1})
    <class 'set'> (length=1)\n  (0): 1

    ```
    """

    def format(
        self, summarizer: BaseSummarizer, value: set, depth: int = 0, max_depth: int = 1
    ) -> str:
        if depth >= max_depth:
            return summarizer.summary(str(value), depth=depth + 1, max_depth=max_depth)
        typ = type(value)
        length = len(value)
        if length > 0:
            if self._max_items >= 0:
                value = islice(value, self._max_items)
            value = str_sequence(
                [summarizer.summary(val, depth=depth + 1, max_depth=max_depth) for val in value],
                num_spaces=self._num_spaces,
            )
            if length > self._max_items and self._max_items > 0:
                value = f"{value}\n..."
            value = f"(length={length:,})\n{value}"
        return str_indent(f"{typ} {value}", num_spaces=self._num_spaces)
