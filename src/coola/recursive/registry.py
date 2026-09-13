r"""Define the transformer registry for recursive data transformation.

This module provides a registry system that manages and dispatches
transformers based on data types, enabling recursive transformation of
nested data structures while preserving their original structure.
"""

from __future__ import annotations

__all__ = ["TransformerRegistry"]

from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer
from coola.registry import BaseTypeDispatchRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


class TransformerRegistry(BaseTypeDispatchRegistry[BaseTransformer[Any]]):
    r"""Registry that manages and dispatches transformers based on data
    type.

    This registry maintains a mapping from Python types to transformer
    instances and uses the Method Resolution Order (MRO) for type
    lookup. When transforming data, it automatically selects the most
    specific registered transformer for the data's type, falling back
    to parent types or a default transformer if needed.

    Type lookups are cached internally for performance. The cache is
    cleared automatically whenever the registry is modified (e.g. via
    ``register`` or ``register_many``).

    Args:
        initial_state: Optional initial mapping of types to transformers.
            If provided, the state is copied to prevent external mutations.

    Example:
        Basic usage with a sequence transformer:

        ```pycon
        >>> from coola.recursive import TransformerRegistry, SequenceTransformer, DefaultTransformer
        >>> registry = TransformerRegistry(
        ...     {object: DefaultTransformer(), list: SequenceTransformer()}
        ... )
        >>> registry
        TransformerRegistry(
          (state): TypeRegistry(
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

    def has_transformer(self, data_type: type) -> bool:
        """Type-specific alias for :meth:`has`: check if a transformer
        is explicitly registered for the given type.

        See :meth:`BaseTypeDispatchRegistry.has` for the full behavior
        description.

        Args:
            data_type: The type to check.

        Returns:
            ``True`` if a transformer is explicitly registered for this type,
                ``False`` otherwise.

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
        return self.has(data_type)

    def find_transformer(self, data_type: type) -> BaseTransformer[Any]:
        """Type-specific alias for :meth:`find`: find the appropriate
        transformer for a given type.

        See :meth:`BaseTypeDispatchRegistry.find` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match).

        Args:
            data_type: The Python type to find a transformer for.

        Returns:
            The most specific registered transformer for this type, resolved
            via MRO, or the default transformer if no match is found.

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
        return self.find(data_type)

    def transform(self, data: object, func: Callable[[Any], Any]) -> Any:
        """Transform data by applying a function recursively through the
        structure.

        This is the main entry point for transformation. It automatically:

        1. Determines the data's type.
        2. Finds the appropriate transformer via ``find_transformer``.
        3. Delegates to that transformer's ``transform`` method, which
           recursively processes any nested structures.

        The original structure of the data is preserved - only the leaf
        values are transformed by the provided function.

        Args:
            data: The data to transform. Can be a nested structure such as
                a ``list``, ``dict``, or ``tuple``.
            func: Function to apply to leaf values. Should accept one argument
                and return a transformed value.

        Returns:
            The transformed data, with the same structure as the input but
            with leaf values transformed by ``func``.

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
