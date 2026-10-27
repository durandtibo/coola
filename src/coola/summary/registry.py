r"""Define the summarizer registry for nested data.

This module provides a registry system that manages and dispatches
summarizers based on data types, enabling recursive summarization of
nested data structures while preserving their original structure.
"""

from __future__ import annotations

__all__ = ["SummarizerRegistry"]

from typing import Any

from coola.registry import BaseTypeDispatchRegistry
from coola.registry.dispatch import FallbackMixin
from coola.summary.base import Summarizer
from coola.summary.config import SummaryConfig
from coola.summary.engine import render


class SummarizerRegistry(FallbackMixin[Summarizer[Any]], BaseTypeDispatchRegistry[Summarizer[Any]]):
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
              (<class 'object'>): DefaultSummarizer()
              (<class 'list'>): SequenceSummarizer()
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
        """Type-specific alias for ``has``: check if a summarizer
        is explicitly registered for the given type.

        See ``BaseTypeDispatchRegistry.has`` for the full behavior
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

    def find_summarizer(self, data_type: type) -> Summarizer[Any]:
        """Type-specific alias for ``find``: find the appropriate
        summarizer for a given type.

        See ``BaseTypeDispatchRegistry.find`` for the full behavior
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
            SequenceSummarizer()

            ```
        """
        return self.find(data_type)

    def summarize(self, data: object, config: SummaryConfig | None = None) -> str:
        r"""Generate a formatted string summary of the provided data.

        The summary is built by the engine (see ``coola.summary.engine``),
        which expands the nodes with the summarizer of their type, up to
        ``config.max_depth`` levels.

        Args:
            data: The data to summarize. Can be any Python object,
                though behavior depends on the registered summarizers.
            config: The options of the summary. If ``None``, the
                default options are used.

        Returns:
            A formatted string representation of the data. The exact
                format depends on the registered summarizers, but it
                includes type information, size metadata, and indented
                content for nested structures.

        Raises:
            KeyError: If no summarizer is registered for a type of the
                data, and not even for ``object``.

        Example:
            ```pycon
            >>> from coola.summary import SummaryConfig, get_default_registry
            >>> registry = get_default_registry()
            >>> print(registry.summarize(1))
            <class 'int'> 1
            >>> print(registry.summarize(["abc", "def"]))
            <class 'list'> (length=2)
              (0): abc
              (1): def
            >>> print(registry.summarize([[0, 1, 2], {"key1": "abc"}]))
            <class 'list'> (length=2)
              (0): [0, 1, 2]
              (1): {'key1': 'abc'}
            >>> print(registry.summarize([[0, 1, 2], {"key1": "abc"}], SummaryConfig(max_depth=2)))
            <class 'list'> (length=2)
              (0): <class 'list'> (length=3)
                (0): 0
                (1): 1
                (2): 2
              (1): <class 'dict'> (length=1)
                (key1): abc

            ```
        """
        return render(data, self, SummaryConfig() if config is None else config)
