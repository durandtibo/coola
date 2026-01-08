r"""Define the transformer registry for recursive data transformation.

This module provides a registry system that manages and dispatches
transformers based on data types, enabling recursive transformation of
nested data structures while preserving their original structure.
"""

from __future__ import annotations

__all__ = ["TransformerRegistry"]

from typing import TYPE_CHECKING

from coola.recursive.base import BaseTransformer
from coola.registry import TypeRegistry
from coola.utils.format import repr_indent, str_indent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class TransformerRegistry:
    """Registry that manages and dispatches transformers based on data
    type.

    This registry maintains a mapping from Python types to transformer instances
    and uses the Method Resolution Order (MRO) for type lookup. When transforming
    data, it automatically selects the most specific registered transformer for
    the data's type, falling back to parent types or a default transformer if needed.

    The registry includes an LRU cache for type lookups to optimize performance
    in applications that repeatedly transform similar data structures.

    Args:
        initial_state: Optional initial mapping of types to transformers.
            If provided, the state is copied to prevent external mutations.

    Attributes:
        _state: Internal mapping of registered types to transformers

    Example:
        Basic usage with a sequence transformer:

        ```pycon
        >>> from coola.recursive import TransformerRegistry, SequenceTransformer, DefaultTransformer
        >>> registry = TransformerRegistry(
        ...     {object: DefaultTransformer(), list: SequenceTransformer()}
        ... )
        >>> registry
        TransformerRegistry(
          TypeRegistry(
            (<class 'object'>): DefaultTransformer()
            (<class 'list'>): SequenceTransformer()
          )
        )
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']

        ```

        Registering custom transformers:

        ```pycon
        >>> from coola.recursive import TransformerRegistry, SequenceTransformer
        >>> registry = TransformerRegistry({object: DefaultTransformer()})
        >>> registry.register(list, SequenceTransformer())
        >>> registry.transform([1, 2, 3], lambda x: x * 2)
        [2, 4, 6]

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.recursive import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> registry.transform(data, str)
        {'a': ['1', '2'], 'b': ['3', '4']}

        ```
    """

    def __init__(self, initial_state: dict[type, BaseTransformer[object]] | None = None) -> None:
        self._state: TypeRegistry[BaseTransformer] = TypeRegistry[BaseTransformer](initial_state)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(self._state)}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(self._state)}\n)"

    def register(
        self,
        data_type: type,
        transformer: BaseTransformer[object],
        exist_ok: bool = False,
    ) -> None:
        """Register a transformer for a given data type.

        This method associates a transformer instance with a specific Python type.
        When data of this type is transformed, the registered transformer will be used.
        The cache is automatically cleared after registration to ensure consistency.

        Args:
            data_type: The Python type to register (e.g., list, dict, custom classes)
            transformer: The transformer instance that handles this type
            exist_ok: If False (default), raises an error if the type is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.recursive import TransformerRegistry, SequenceTransformer
            >>> registry = TransformerRegistry()
            >>> registry.register(list, SequenceTransformer())
            >>> registry.has_transformer(list)
            True

            ```
        """
        self._state.register(data_type, transformer, exist_ok=exist_ok)

    def register_many(
        self,
        mapping: Mapping[type, BaseTransformer[object]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple transformers at once.

        This is a convenience method for bulk registration that internally calls
        register() for each type-transformer pair.

        Args:
            mapping: Dictionary mapping Python types to transformer instances
            exist_ok: If False (default), raises an error if any type is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.recursive import TransformerRegistry, SequenceTransformer, MappingTransformer
            >>> registry = TransformerRegistry()
            >>> registry.register_many(
            ...     {
            ...         list: SequenceTransformer(),
            ...         dict: MappingTransformer(),
            ...     }
            ... )
            >>> registry
            TransformerRegistry(
              TypeRegistry(
                (<class 'list'>): SequenceTransformer()
                (<class 'dict'>): MappingTransformer()
              )
            )

            ```
        """
        self._state.register_many(mapping, exist_ok=exist_ok)

    def has_transformer(self, data_type: type) -> bool:
        """Check if a transformer is explicitly registered for the given
        type.

        Note that this only checks for direct registration. Even if this returns
        False, find_transformer() may still return a transformer via MRO lookup
        or the default transformer.

        Args:
            data_type: The type to check

        Returns:
            True if a transformer is explicitly registered for this type,
            False otherwise

        Example:
            ```pycon
            >>> from coola.recursive import TransformerRegistry, SequenceTransformer
            >>> registry = TransformerRegistry()
            >>> registry.register(list, SequenceTransformer())
            >>> registry.has_transformer(list)
            True
            >>> registry.has_transformer(tuple)
            False

            ```
        """
        return data_type in self._state

    def find_transformer(self, data_type: type) -> BaseTransformer[object]:
        """Find the appropriate transformer for a given type.

        Uses the Method Resolution Order (MRO) to find the most specific
        registered transformer. For example, if you register a transformer
        for Sequence but not for list, lists will use the Sequence transformer.

        Results are cached using an LRU cache (256 entries) for performance,
        as transformer lookup is a hot path in recursive transformations.

        Args:
            data_type: The Python type to find a transformer for

        Returns:
            The most specific registered transformer for this type, a parent
            type's transformer via MRO, or the default transformer

        Example:
            ```pycon
            >>> from collections.abc import Sequence
            >>> from coola.recursive import TransformerRegistry, SequenceTransformer, DefaultTransformer
            >>> registry = TransformerRegistry({object: DefaultTransformer()})
            >>> registry.register(Sequence, SequenceTransformer())
            >>> # list does not inherit from Sequence, so it uses DefaultTransformer
            >>> transformer = registry.find_transformer(list)
            >>> transformer
            DefaultTransformer()

            ```
        """
        return self._state.resolve(data_type)

    def transform(self, data: object, func: Callable[[object], object]) -> object:
        """Transform data by applying a function recursively through the
        structure.

        This is the main entry point for transformation. It automatically:
        1. Determines the data's type
        2. Finds the appropriate transformer
        3. Delegates to that transformer's transform method
        4. The transformer recursively processes nested structures

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
            >>> from coola.recursive import get_default_registry
            >>> registry = get_default_registry()
            >>> registry.transform({"scores": [95, 87, 92], "name": "test"}, str)
            {'scores': ['95', '87', '92'], 'name': 'test'}

            ```

            Doubling all numeric values:

            ```pycon
            >>> from coola.recursive import get_default_registry
            >>> registry = get_default_registry()
            >>> registry.transform(
            ...     [1, [2, 3], {"a": 4}], lambda x: x * 2 if isinstance(x, (int, float)) else x
            ... )
            [2, [4, 6], {'a': 8}]

            ```
        """
        transformer = self.find_transformer(type(data))
        return transformer.transform(data, func, self)
