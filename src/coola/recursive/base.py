r"""Define the transformer base class."""

from __future__ import annotations

__all__ = ["BaseTransformer", "Expansion", "Transformer"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)

# The children of a container and the function that rebuilds the
# container from the transformed children.
Expansion = tuple["Sequence[Any]", "Callable[[list[Any]], Any]"]


class Transformer(Protocol[T_contra]):
    r"""Structural interface of a type-specific transformer.

    Any object with a compatible ``expand`` method can be registered
    without subclassing ``BaseTransformer``, which remains available as
    a convenience base class.
    """

    def expand(self, data: T_contra) -> Expansion | None:
        r"""Describe the children of ``data`` and how to rebuild it.

        Args:
            data: The data to expand.

        Returns:
            ``None`` if ``data`` is a leaf. Otherwise, a tuple
            ``(children, rebuild)``: the children of ``data`` and the
            function that rebuilds ``data`` from a list with the
            transformed children.
        """


class BaseTransformer(ABC, Generic[T]):
    r"""Base class for type-specific transformers.

    A transformer only describes the shape of a type: its children and
    how to rebuild it from the transformed children. The engine
    (``coola.recursive.engine``) walks the children, applies the function
    to the leaves and detects the cycles, so a transformer cannot forget
    a protection and does not recurse.

    The contract of ``expand``:

    1. It is pure and cheap: it does not call the function, does not call
        the registry, does not recurse and does not modify ``data``.
    2. ``rebuild`` receives a new ``list`` with the transformed children,
        in the order of ``children``, and returns a new container (it
        returns ``data`` itself only if it has no children and must be
        kept as it is, like ``IdentityTransformer``).
    3. A container with no child (``[]``, ``{}``) returns empty
        ``children``: ``rebuild([])`` builds an empty container of the
        same type.
    4. A leaf returns ``None``: the engine calls the function on it.

    Notes:
        Subclasses must implement the `expand` method.

    Example:
        ```pycon
        >>> from coola.recursive import SequenceTransformer
        >>> transformer = SequenceTransformer()
        >>> transformer
        SequenceTransformer()
        >>> children, rebuild = transformer.expand([1, 2, 3])
        >>> children
        [1, 2, 3]
        >>> rebuild(["1", "2", "3"])
        ['1', '2', '3']

        ```
    """

    @abstractmethod
    def expand(self, data: T) -> Expansion | None:
        r"""Describe the children of ``data`` and how to rebuild it.

        Args:
            data: The data to expand. Must be of type T that this
                transformer handles.

        Returns:
            ``None`` if ``data`` is a leaf. Otherwise, a tuple
            ``(children, rebuild)``: the children of ``data`` and the
            function that rebuilds ``data`` from a list with the
            transformed children.

        Example:
            ```pycon
            >>> from coola.recursive import DefaultTransformer, MappingTransformer
            >>> DefaultTransformer().expand(42) is None
            True
            >>> children, rebuild = MappingTransformer().expand({"a": 1, "b": 2})
            >>> children
            [1, 2]
            >>> rebuild([10, 20])
            {'a': 10, 'b': 20}

            ```
        """
