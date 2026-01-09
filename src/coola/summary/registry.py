r"""Define the summarizer registry for nested data.

This module provides a registry system that manages and dispatches
summarizers based on data types, enabling recursive summarization of
nested data structures while preserving their original structure.
"""

from __future__ import annotations

__all__ = ["SummarizerRegistry"]

from typing import TYPE_CHECKING, Any

from coola.registry import TypeRegistry
from coola.summary.base import BaseSummarizer
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# TODO(tibo): update docstrings and examples


class SummarizerRegistry:
    """Registry that manages and dispatches summarizers based on data
    type.

    This registry maintains a mapping from Python types to summarizer instances
    and uses the Method Resolution Order (MRO) for type lookup. When transforming
    data, it automatically selects the most specific registered summarizer for
    the data's type, falling back to parent types or a default summarizer if needed.

    The registry includes an LRU cache for type lookups to optimize performance
    in applications that repeatedly transform similar data structures.

    Args:
        initial_state: Optional initial mapping of types to summarizers.
            If provided, the state is copied to prevent external mutations.

    Attributes:
        _state: Internal mapping of registered types to summarizers

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
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']

        ```

        Registering custom summarizers:

        ```pycon
        >>> from coola.summary import SummarizerRegistry, SequenceSummarizer
        >>> registry = SummarizerRegistry({object: DefaultSummarizer()})
        >>> registry.register(list, SequenceSummarizer())
        >>> registry.transform([1, 2, 3], lambda x: x * 2)
        [2, 4, 6]

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.summary import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> registry.transform(data, str)
        {'a': ['1', '2'], 'b': ['3', '4']}

        ```
    """

    def __init__(self, initial_state: dict[type, BaseSummarizer[Any]] | None = None) -> None:
        self._state: TypeRegistry[BaseSummarizer] = TypeRegistry[BaseSummarizer](initial_state)

    def __repr__(self) -> str:
        state = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

    def __str__(self) -> str:
        state = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

    def register(
        self,
        data_type: type,
        summarizer: BaseSummarizer[Any],
        exist_ok: bool = False,
    ) -> None:
        """Register a summarizer for a given data type.

        This method associates a summarizer instance with a specific Python type.
        When data of this type is transformed, the registered summarizer will be used.
        The cache is automatically cleared after registration to ensure consistency.

        Args:
            data_type: The Python type to register (e.g., list, dict, custom classes)
            summarizer: The summarizer instance that handles this type
            exist_ok: If False (default), raises an error if the type is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.summary import SummarizerRegistry, SequenceSummarizer
            >>> registry = SummarizerRegistry()
            >>> registry.register(list, SequenceSummarizer())
            >>> registry.has_summarizer(list)
            True

            ```
        """
        self._state.register(data_type, summarizer, exist_ok=exist_ok)

    def register_many(
        self,
        mapping: Mapping[type, BaseSummarizer[Any]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple summarizers at once.

        This is a convenience method for bulk registration that internally calls
        register() for each type-summarizer pair.

        Args:
            mapping: Dictionary mapping Python types to summarizer instances
            exist_ok: If False (default), raises an error if any type is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.summary import SummarizerRegistry, SequenceSummarizer, MappingSummarizer
            >>> registry = SummarizerRegistry()
            >>> registry.register_many(
            ...     {
            ...         list: SequenceSummarizer(),
            ...         dict: MappingSummarizer(),
            ...     }
            ... )
            >>> registry
            SummarizerRegistry(
              (state): TypeRegistry(
                  (<class 'list'>): SequenceSummarizer()
                  (<class 'dict'>): MappingSummarizer()
                )
            )

            ```
        """
        self._state.register_many(mapping, exist_ok=exist_ok)

    def has_summarizer(self, data_type: type) -> bool:
        """Check if a summarizer is explicitly registered for the given
        type.

        Note that this only checks for direct registration. Even if this returns
        False, find_summarizer() may still return a summarizer via MRO lookup
        or the default summarizer.

        Args:
            data_type: The type to check

        Returns:
            True if a summarizer is explicitly registered for this type,
            False otherwise

        Example:
            ```pycon
            >>> from coola.summary import SummarizerRegistry, SequenceSummarizer
            >>> registry = SummarizerRegistry()
            >>> registry.register(list, SequenceSummarizer())
            >>> registry.has_summarizer(list)
            True
            >>> registry.has_summarizer(tuple)
            False

            ```
        """
        return data_type in self._state

    def find_summarizer(self, data_type: type) -> BaseSummarizer[Any]:
        """Find the appropriate summarizer for a given type.

        Uses the Method Resolution Order (MRO) to find the most specific
        registered summarizer. For example, if you register a summarizer
        for Sequence but not for list, lists will use the Sequence summarizer.

        Results are cached using an LRU cache (256 entries) for performance,
        as summarizer lookup is a hot path in recursive transformations.

        Args:
            data_type: The Python type to find a summarizer for

        Returns:
            The most specific registered summarizer for this type, a parent
            type's summarizer via MRO, or the default summarizer

        Example:
            ```pycon
            >>> from collections.abc import Sequence
            >>> from coola.summary import SummarizerRegistry, SequenceSummarizer, DefaultSummarizer
            >>> registry = SummarizerRegistry({object: DefaultSummarizer()})
            >>> registry.register(Sequence, SequenceSummarizer())
            >>> # list does not inherit from Sequence, so it uses DefaultSummarizer
            >>> summarizer = registry.find_summarizer(list)
            >>> summarizer
            DefaultSummarizer()

            ```
        """
        return self._state.resolve(data_type)

    def summary(self, data: Any, func: Callable[[Any], Any]) -> Any:
        """Transform data by applying a function recursively through the
        structure.

        This is the main entry point for transformation. It automatically:
        1. Determines the data's type
        2. Finds the appropriate summarizer
        3. Delegates to that summarizer's transform method
        4. The summarizer recursively processes nested structures

        The original structure of the data is preserved - only the leaf values
        are transformed by the provided function.

        Args:
            data: The data to transform (can be nested: lists, dicts, tuples, etc.)
            func: Function to apply to leaf values. Should accept one argument
                and return a transformed value.

        Returns:
            Transformed data with the same structure as the input but with
            leaf values transformed by func

        Example:
            Converting all numbers to strings in a nested structure:

            ```pycon
            >>> from coola.summary import get_default_registry
            >>> registry = get_default_registry()
            >>> registry.transform({"scores": [95, 87, 92], "name": "test"}, str)
            {'scores': ['95', '87', '92'], 'name': 'test'}

            ```

            Doubling all numeric values:

            ```pycon
            >>> from coola.summary import get_default_registry
            >>> registry = get_default_registry()
            >>> registry.transform(
            ...     [1, [2, 3], {"a": 4}], lambda x: x * 2 if isinstance(x, (int, float)) else x
            ... )
            [2, [4, 6], {'a': 8}]

            ```
        """
        summarizer = self.find_summarizer(type(data))
        return summarizer.transform(data, func, self)
