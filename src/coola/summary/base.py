r"""Define the summarizer base class.

This module provides the ``Node`` that a summarizer returns for a
container, the ``Summarizer`` protocol and an abstract base class for
implementing summarizers that describe one type. A summarizer only
describes its type: the recursion, the depth, the limits, the cycles and
the errors are handled by the engine (see ``coola.summary.engine``).
"""

from __future__ import annotations

__all__ = ["BaseSummarizer", "Node", "Summarizer"]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

    from coola.summary.config import SummaryConfig

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


@dataclass(slots=True)
class Node:
    r"""Describe a container for the engine.

    Args:
        header: The first line of the node, for example
            ``"<class 'list'> (length=3)"``.
        children: The ``(label, child)`` pairs, in display order. It
            should be lazy (a generator, ``enumerate(data)``,
            ``data.items()``): the engine takes at most ``max_items``
            of them.
        total: The number of children, used to announce
            ``... (+N more)`` when some of them are not shown.

    Example:
        ```pycon
        >>> from coola.summary import Node
        >>> node = Node("<class 'list'> (length=2)", enumerate(["a", "b"]), 2)
        >>> node.header
        "<class 'list'> (length=2)"
        >>> list(node.children)
        [(0, 'a'), (1, 'b')]
        >>> node.total
        2

        ```
    """

    header: str
    children: Iterable[tuple[Any, Any]]
    total: int


class Summarizer(Protocol[T_contra]):
    r"""Structural interface of a type-specific summarizer.

    Any object with a compatible ``summarize`` method can be registered
    without subclassing ``BaseSummarizer``, which remains available as a
    convenience base class.
    """

    def summarize(self, data: T_contra, config: SummaryConfig, *, compact: bool) -> str | Node:
        r"""Summarize the data.

        Args:
            data: The data to summarize.
            config: The options of the summary.
            compact: If ``True``, the node is not expanded, so a
                summary on one line is enough.

        Returns:
            A ``str`` for a leaf, or a ``Node`` (a header and lazy
            children) for a container.
        """


class BaseSummarizer(ABC, Generic[T]):
    r"""Abstract base class for implementing data summarizers.

    A summarizer describes one type: it returns a ``str`` for a leaf, or
    a ``Node`` (a header and lazy children) for a container. The engine
    recurses into the children, so a summarizer does not call the
    registry.

    Notes:
        The rules of the contract:

        1. ``summarize`` is pure and cheap: it does not call the
            registry, does not recurse, does not modify ``data`` and does
            not iterate the whole data. For a container, it reads
            ``len(data)`` and returns the children lazily. A set is
            the exception: it returns its smallest ``max_items`` items.
        2. ``compact`` is ``True`` when the data is at the display
            depth. A leaf may use it. A container ignores it: the
            engine shows it on one line.
        3. A leaf returns a ``str``, which may have several lines. The
            engine cuts it at ``max_characters``.
        4. The labels of the children can be any object: the engine
            writes their ``str`` cut at ``max_characters``.
        5. An exception (of ``summarize``, of the iteration of the
            children, of the ``str`` of a label) is handled by the
            engine, which writes a placeholder. The summarizer does not
            guard itself.

    Example:
        ```pycon
        >>> from coola.summary import DefaultSummarizer, SummaryConfig
        >>> summarizer = DefaultSummarizer()
        >>> print(summarizer.summarize(1, SummaryConfig(), compact=False))
        <class 'int'> 1

        ```
    """

    def __repr__(self) -> str:
        r"""Return the representation of the summarizer.

        Returns:
            The class name followed by parentheses.
        """
        return f"{type(self).__qualname__}()"

    def equal(self, other: object) -> bool:
        r"""Check equality between this summarizer and another object.

        Two summarizers are equal if they have the exact same type: the
        built-in summarizers are stateless. A summarizer with a state
        should override this method.

        Args:
            other: The object to compare with this summarizer.

        Returns:
            ``True`` if the objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.summary import DefaultSummarizer, MappingSummarizer
            >>> DefaultSummarizer().equal(DefaultSummarizer())
            True
            >>> DefaultSummarizer().equal(MappingSummarizer())
            False

            ```
        """
        return type(other) is type(self)

    @abstractmethod
    def summarize(self, data: T, config: SummaryConfig, *, compact: bool) -> str | Node:
        r"""Describe the data.

        Args:
            data: The data to describe.
            config: The options of the summary.
            compact: ``True`` if the data is at the display depth, so
                it is shown on one line.

        Returns:
            A ``str`` for a leaf, or a ``Node`` for a container.

        Example:
            ```pycon
            >>> from coola.summary import SequenceSummarizer, SummaryConfig
            >>> node = SequenceSummarizer().summarize([1, 2], SummaryConfig(), compact=False)
            >>> node.header
            "<class 'list'> (length=2)"

            ```
        """
