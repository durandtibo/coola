r"""Define the summarizer base class.

This module provides an abstract base class for implementing summarizers
that convert Python objects into human-readable string representations.
Summarizers are particularly useful for inspecting nested data
structures with configurable depth limits.
"""

from __future__ import annotations

__all__ = ["BaseSummarizer"]


from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry

T = TypeVar("T")


class BaseSummarizer(ABC, Generic[T]):
    r"""Abstract base class for implementing data summarizers.

    A summarizer converts Python objects into formatted string representations,
    with support for nested structures and configurable depth limits. This is
    useful for debugging, logging, and displaying complex data in a readable format.

    The class is generic over type T, allowing concrete implementations to
    specialize for specific data types while maintaining type safety.

    Notes:
        Concrete implementations must override the ``summary`` method to define
        how data should be formatted and displayed.

        The depth mechanism allows for progressive disclosure of nested structures,
        preventing overwhelming output for deeply nested data.

    Example:
        ```pycon
        >>> from coola.summary import Summarizer
        >>> summarizer = Summarizer()
        >>> summarizer
        Summarizer()

        >>> # Summarize a simple integer
        >>> print(summarizer.summary(1))
        <class 'int'> 1

        >>> # Summarize a list (shallow, default max_depth=1)
        >>> print(summarizer.summary(["abc", "def"]))
        <class 'list'> (length=2)
          (0): abc
          (1): def

        >>> # Nested structures shown compactly at depth 1
        >>> print(summarizer.summary([[0, 1, 2], {"key1": "abc", "key2": "def"}]))
        <class 'list'> (length=2)
          (0): [0, 1, 2]
          (1): {'key1': 'abc', 'key2': 'def'}

        >>> # Increase max_depth to expand nested structures
        >>> print(summarizer.summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2))
        <class 'list'> (length=2)
          (0): <class 'list'> (length=3)
              (0): 0
              (1): 1
              (2): 2
          (1): <class 'dict'> (length=2)
              (key1): abc
              (key2): def

        ```
    """

    @abstractmethod
    def summary(
        self,
        data: T,
        registry: SummarizerRegistry,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        r"""Generate a formatted string summary of the provided data.

        This method creates a human-readable representation of the input data,
        with support for nested structures up to a specified depth. When the
        current depth exceeds max_depth, nested structures are typically shown
        in a compact form without further expansion.

        Args:
            data: The data object to summarize. Can be any Python object,
                though behavior depends on the concrete implementation.
            registry: The summarizer registry used to look up summarizers
                for nested data structures of different types.
            depth: The current nesting level in the data structure. Used
                internally during recursive summarization. Typically starts
                at 0 for top-level calls. Must be non-negative.
            max_depth: The maximum nesting level to expand when summarizing.
                Structures deeper than this level are shown in compact form.
                Must be non-negative. Default is 1, which expands only the
                top level of nested structures.

        Returns:
            A formatted string representation of the data. The exact format
            depends on the concrete implementation, but typically includes
            type information, size/length metadata, and indented content for
            nested structures.

        Raises:
            The base class doesn't specify exceptions, but implementations
            may raise ValueError for invalid depth parameters or other
            exceptions based on the data type being summarized.

        Notes:
            - The depth parameter is primarily for internal use during recursion.
              Most external callers should use the default value of 0.
            - Setting max_depth=0 typically shows only top-level information
              without expanding any nested structures.
            - Higher max_depth values provide more detail but can produce
              very long output for deeply nested data.

        Example:
            ```pycon
            >>> from coola.summary import Summarizer
            >>> summarizer = Summarizer()

            >>> # Simple value
            >>> print(summarizer.summary(1))
            <class 'int'> 1

            >>> # List with default depth (expands first level only)
            >>> print(summarizer.summary(["abc", "def"]))
            <class 'list'> (length=2)
              (0): abc
              (1): def

            >>> # Nested list, default max_depth=1 (inner list not expanded)
            >>> print(summarizer.summary([[0, 1, 2], {"key1": "abc", "key2": "def"}]))
            <class 'list'> (length=2)
              (0): [0, 1, 2]
              (1): {'key1': 'abc', 'key2': 'def'}

            >>> # Nested list with max_depth=2 (expands both levels)
            >>> print(summarizer.summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2))
            <class 'list'> (length=2)
              (0): <class 'list'> (length=3)
                  (0): 0
                  (1): 1
                  (2): 2
              (1): <class 'dict'> (length=2)
                  (key1): abc
                  (key2): def

            >>> # Control depth for very nested structures
            >>> deeply_nested = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
            >>> print(summarizer.summary(deeply_nested, max_depth=1))
            <class 'list'> (length=2)
              (0): [[1, 2], [3, 4]]
              (1): [[5, 6], [7, 8]]

            ```
        """
