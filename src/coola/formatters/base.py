r"""Define the formatter base class."""

from __future__ import annotations

__all__ = ["BaseFormatter"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from coola.summarizers.base import BaseSummarizer

T = TypeVar("T")


class BaseFormatter(ABC, Generic[T]):
    r"""Define the base class to implement a formatter.

    Example usage:

    ```pycon

    >>> from coola import Summarizer
    >>> from coola.formatters import DefaultFormatter
    >>> formatter = DefaultFormatter()
    >>> formatter.format(Summarizer(), 1)
    <class 'int'> 1

    ```
    """

    @abstractmethod
    def clone(self) -> BaseFormatter:
        r"""Return a copy of the formatter.

        Returns:
            A copy of the formatter.

        Example usage:

        ```pycon
        >>> from coola.formatters import DefaultFormatter
        >>> formatter = DefaultFormatter()
        >>> formatter2 = formatter.clone()
        >>> formatter.set_max_characters(10)
        >>> formatter
        DefaultFormatter(max_characters=10)
        >>> formatter2
        DefaultFormatter(max_characters=-1)

        ```
        """

    @abstractmethod
    def equal(self, other: Any) -> bool:
        r"""Indicate if the other object is equal to the self object.

        Args:
            other: The other object to compare.

        Returns:
            ``True`` if the objects are equal, otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.formatters import DefaultFormatter
        >>> formatter = DefaultFormatter()
        >>> formatter.equal(DefaultFormatter())
        True
        >>> formatter.equal(DefaultFormatter(max_characters=10))
        False

        ```
        """

    @abstractmethod
    def format(self, summarizer: BaseSummarizer, value: T, depth: int, max_depth: int) -> str:
        r"""Format a value.

        Args:
            summarizer: The summarizer.
            value: The value to summarize.
            depth: The current depth.
            max_depth: The maximum depth.

        Returns:
            str: The formatted value.

        Example usage:

        ```pycon
        >>> from coola import Summarizer
        >>> from coola.formatters import DefaultFormatter
        >>> formatter = DefaultFormatter()
        >>> formatter.format(Summarizer(), 1)
        <class 'int'> 1

        ```
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Load the state values from a dict.

        Args:
            state_dict: a dict with parameters

        Example usage:

        ```pycon
        >>> from coola.formatters import DefaultFormatter
        >>> formatter = DefaultFormatter()
        >>> # Please take a look to the implementation of the state_dict
        >>> # function to know the expected structure
        >>> formatter.load_state_dict({"max_characters": 10})
        >>> formatter
        DefaultFormatter(max_characters=10)

        ```
        """

    @abstractmethod
    def state_dict(self) -> dict:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon
        >>> from coola.formatters import DefaultFormatter
        >>> formatter = DefaultFormatter()
        >>> formatter.state_dict()
        {'max_characters': -1}

        ```
        """
