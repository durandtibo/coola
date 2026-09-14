r"""Define the summarizer registry for nested data.

This module provides a registry system that manages and dispatches
summarizers based on data types, enabling recursive summarization of
nested data structures while preserving their original structure.
"""

from __future__ import annotations

__all__ = ["SummarizerRegistry"]

from typing import Any

from coola.registry import BaseTypeDispatchRegistry
from coola.summary.base import BaseSummarizer


class SummarizerRegistry(BaseTypeDispatchRegistry[BaseSummarizer[Any]]):
    r"""Registry that manages and dispatches summarizers based on data
    type.

    This registry maintains a mapping from Python types to summarizer
    instances and uses the Method Resolution Order (MRO) for type
    lookup. When summarizing data, it automatically selects the most
    specific registered summarizer for the data's type, falling back to
    parent types or a default summarizer if needed.

    Type lookups are cached internally for performance. The cache is
    cleared automatically whenever the registry is modified (e.g. via
    ``register`` or ``register_many``).

    Args:
        initial_state: Optional initial mapping of types to summarizers.
            If provided, the state is copied to prevent external mutations.

    Example:
        Basic usage with a sequence summarizer:

        ```pycon
        >>> from coola.summary import SummarizerRegistry, SequenceSummarizer, DefaultSummarizer
        >>> registry = SummarizerRegistry({object: DefaultSummarizer(), list: SequenceSummarizer()})
        >>> registry
        SummarizerRegistry(
          (state): TypeRegistry(
              (<class 'object'>): DefaultSummarizer(max_characters=-1)
              (<class 'list'>): SequenceSummarizer(max_items=5, num_spaces=2)
            )
        )
        >>> print(registry.summarize([1, 2, 3]))
        <class 'list'> (length=3)
          (0): 1
          (1): 2
          (2): 3

        ```

        Registering custom summarizers:

        ```pycon
        >>> from coola.summary import SummarizerRegistry, SequenceSummarizer
        >>> registry = SummarizerRegistry({object: DefaultSummarizer()})
        >>> registry.register(tuple, SequenceSummarizer())
        >>> print(registry.summarize((1, 2, 3)))
        <class 'tuple'> (length=3)
          (0): 1
          (1): 2
          (2): 3

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.summary import get_default_registry
        >>> registry = get_default_registry()
        >>> print(registry.summarize({"a": [1, 2], "b": [3, 4]}))
        <class 'dict'> (length=2)
          (a): [1, 2]
          (b): [3, 4]

        ```
    """

    def has_summarizer(self, data_type: type) -> bool:
        """Type-specific alias for :meth:`has`: check if a summarizer
        is explicitly registered for the given type.

        See :meth:`BaseTypeDispatchRegistry.has` for the full behavior
        description.

        Args:
            data_type: The type to check.

        Returns:
            ``True`` if a summarizer is explicitly registered for this type,
                ``False`` otherwise.

        Example:
            ```pycon
            >>> from coola.summary import SummarizerRegistry, SequenceSummarizer
            >>> registry = SummarizerRegistry({list: SequenceSummarizer()})
            >>> registry.has_summarizer(list)
            True
            >>> registry.has_summarizer(tuple)
            False

            ```
        """
        return self.has(data_type)

    def find_summarizer(self, data_type: type) -> BaseSummarizer[Any]:
        """Type-specific alias for :meth:`find`: find the appropriate
        summarizer for a given type.

        See :meth:`BaseTypeDispatchRegistry.find` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match).

        Args:
            data_type: The Python type to find a summarizer for.

        Returns:
            The most specific registered summarizer for this type, resolved
            via MRO, or the default summarizer if no match is found.

        Example:
            ```pycon
            >>> from coola.summary import get_default_registry
            >>> registry = get_default_registry()
            >>> summarizer = registry.find_summarizer(list)
            >>> summarizer
            SequenceSummarizer(max_items=5, num_spaces=2)

            ```
        """
        return self.find(data_type)

    def summarize(self, data: object, depth: int = 0, max_depth: int = 1) -> str:
        r"""Generate a formatted string summary of the provided data.

        This method creates a human-readable representation of the input data,
        with support for nested structures up to a specified depth. When the
        current depth exceeds max_depth, nested structures are typically shown
        in a compact form without further expansion.

        Args:
            data: The data object to summarize. Can be any Python object,
                though behavior depends on the registered summarizers.
            depth: The current nesting level in the data structure. Used
                internally during recursive summarization. Typically starts
                at 0 for top-level calls. Must be non-negative.
            max_depth: The maximum nesting level to expand when summarizing.
                Structures deeper than this level are shown in compact form.
                Must be non-negative. Default is 1, which expands only the
                top level of nested structures.

        Returns:
            A formatted string representation of the data. The exact format
            depends on the registered summarizer, but typically includes
            type information, size/length metadata, and indented content for
            nested structures.

        Raises:
            ValueError: May be raised by individual summarizers for invalid
                depth parameters or other issues based on the data type being
                summarized. The registry itself doesn't raise exceptions, but
                delegates to registered summarizers which may raise this or
                other exceptions.

        Notes:
            - The depth parameter is primarily for internal use during recursion.
              Most external callers should use the default value of 0.
            - Setting max_depth=0 typically shows only top-level information
              without expanding any nested structures.
            - Higher max_depth values provide more detail but can produce
              very long output for deeply nested data.

        Example:
            ```pycon
            >>> from coola.summary import get_default_registry
            >>> registry = get_default_registry()

            >>> # Simple value
            >>> print(registry.summarize(1))
            <class 'int'> 1

            >>> # List with default depth (expands first level only)
            >>> print(registry.summarize(["abc", "def"]))
            <class 'list'> (length=2)
              (0): abc
              (1): def

            >>> # Nested list, default max_depth=1 (inner list not expanded)
            >>> print(registry.summarize([[0, 1, 2], {"key1": "abc", "key2": "def"}]))
            <class 'list'> (length=2)
              (0): [0, 1, 2]
              (1): {'key1': 'abc', 'key2': 'def'}

            >>> # Nested list with max_depth=2 (expands both levels)
            >>> print(registry.summarize([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2))
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
            >>> print(registry.summarize(deeply_nested))
            <class 'list'> (length=2)
              (0): [[1, 2], [3, 4]]
              (1): [[5, 6], [7, 8]]

            ```
        """
        summarizer = self.find_summarizer(type(data))
        return summarizer.summarize(data=data, depth=depth, max_depth=max_depth, registry=self)
