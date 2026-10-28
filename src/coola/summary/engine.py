r"""Implement the engine that renders a summary.

The engine owns the invariants of the package: bounded output, bounded
work, display depth, cycles, totality (an error in the data is shown as
a placeholder) and layout. A summarizer only describes one type (see
``coola.summary.base``), so it cannot forget a protection and does not
recurse: the engine walks the nodes with an explicit stack, so there is
no ``RecursionError``, whatever the depth.
"""

from __future__ import annotations

__all__ = ["CompactRepr", "render"]

import sys
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any

from coola.summary.base import Node
from coola.utils.safe import safe_repr, safe_str
from coola.utils.selection import smallest_items

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.summary.config import SummaryConfig
    from coola.summary.registry import SummarizerRegistry

INDENT = "  "

_COMPACT_LEVELS = 3
_FILL = "..."
_LITERALS = (int, float, bool, type(None))


def _cut(text: str, max_characters: int | None) -> str:
    r"""Cut a text to ``max_characters`` characters, followed by ``...``
    if it is cut."""
    if max_characters is None or len(text) <= max_characters:
        return text
    return text[:max_characters] + "..."


_METHODS = {
    str: "_repr_str",
    list: "_repr_list",
    tuple: "_repr_tuple",
    dict: "_repr_dict",
    deque: "_repr_deque",
    set: "_repr_set",
    frozenset: "_repr_frozenset",
}


class CompactRepr:
    r"""Build the one-line form of a container that is at the display
    depth.

    The work is bounded by ``max_items`` and by three levels of nesting,
    whatever the size of the container: ``list``, ``tuple``, ``deque``,
    ``dict``, ``set`` and ``frozenset`` are cut at ``max_items`` items,
    and a string is shortened in the middle when it is longer than
    ``max_characters``. A dict keeps its order, and a set shows its
    smallest items (as ``SetSummarizer``). Any other type (a subclass of
    ``list``, a ``Counter``, an array, ...) is shown with its ``repr``
    cut at ``max_characters``, which is computed in full.

    Args:
        config: The options of the summary.

    Example:
        ```pycon
        >>> from coola.summary import SummaryConfig
        >>> from coola.summary.engine import CompactRepr
        >>> compact = CompactRepr(SummaryConfig(max_items=3))
        >>> compact.repr(list(range(10)))
        '[0, 1, 2, ...]'
        >>> compact.repr({"b": 1, "a": 2})
        "{'b': 1, 'a': 2}"
        >>> compact.repr([[[[1]]]])
        '[[[[...]]]]'

        ```
    """

    def __init__(self, config: SummaryConfig) -> None:
        self._max_items = sys.maxsize if config.max_items is None else config.max_items
        self._max_characters = config.max_characters

    def repr(self, data: Any) -> str:
        r"""Return the one-line form of the data.

        Args:
            data: The data to show on one line.

        Returns:
            The one-line form.
        """
        return self._repr(data, _COMPACT_LEVELS)

    def _repr(self, data: Any, level: int) -> str:
        kind = type(data)
        if kind in _LITERALS:
            return repr(data)
        method = _METHODS.get(kind)
        if method is None:
            return _cut(safe_repr(data), self._max_characters)
        return getattr(self, method)(data, level)

    def _repr_str(self, data: str, level: int) -> str:  # noqa: ARG002
        limit = self._max_characters
        if limit is None or len(data) <= limit:
            return repr(data)
        head = max(0, (limit - 3) // 2)
        tail = max(0, limit - 3 - head)
        text = repr(data[:head] + data[len(data) - tail :])
        return text[:head] + _FILL + text[len(text) - tail :]

    def _repr_list(self, data: list[Any], level: int) -> str:
        return self._repr_items(data, level, "[", "]")

    def _repr_tuple(self, data: tuple[Any, ...], level: int) -> str:
        return self._repr_items(data, level, "(", ",)" if len(data) == 1 else ")")

    def _repr_deque(self, data: deque[Any], level: int) -> str:
        return self._repr_items(data, level, "deque([", "])")

    def _repr_set(self, data: set[Any], level: int) -> str:
        return self._repr_sorted(data, level, "{", "}", "set()")

    def _repr_frozenset(self, data: frozenset[Any], level: int) -> str:
        return self._repr_sorted(data, level, "frozenset({", "})", "frozenset()")

    def _repr_items(self, data: Any, level: int, left: str, right: str) -> str:
        if not data:
            return left + right.replace(",", "")
        if level <= 0:
            return left + _FILL + right.replace(",", "")
        pieces = [self._repr(item, level - 1) for item in islice(data, self._max_items)]
        if len(data) > self._max_items:
            pieces.append(_FILL)
        return left + ", ".join(pieces) + right

    def _repr_dict(self, data: dict[Any, Any], level: int) -> str:
        if not data:
            return "{}"
        if level <= 0:
            return "{...}"
        pieces = [
            f"{self._repr(key, level - 1)}: {self._repr(value, level - 1)}"
            for key, value in islice(data.items(), self._max_items)
        ]
        if len(data) > self._max_items:
            pieces.append(_FILL)
        return "{" + ", ".join(pieces) + "}"

    def _repr_sorted(self, data: Any, level: int, left: str, right: str, empty: str) -> str:
        if not data:
            return empty
        if level <= 0:
            return left + _FILL + right
        pieces = [
            self._repr(item, level - 1) for item in smallest_items(data, limit=self._max_items)
        ]
        if len(data) > self._max_items:
            pieces.append(_FILL)
        return left + ", ".join(pieces) + right


@dataclass(slots=True)
class _Frame:
    r"""The state of a node that is being expanded."""

    data: Any
    children: Iterator[tuple[Any, Any]]
    depth: int
    total: int
    shown: int = 0


class _Renderer:
    r"""Walk the nodes with an explicit stack and write the lines."""

    def __init__(self, registry: SummarizerRegistry, config: SummaryConfig) -> None:
        self._registry = registry
        self._config = config
        self._lines: list[str] = []
        self._ancestors: set[int] = set()  # the ids of the nodes that are being expanded
        self._stack: list[_Frame] = []

    def render(self, data: object) -> str:
        self._visit(None, data, 0)
        while self._stack:
            self._advance(self._stack[-1])
        return "\n".join(self._lines)

    def _advance(self, frame: _Frame) -> None:
        r"""Visit the next child of the frame, or close the frame."""
        try:
            label, child = next(frame.children)
        except StopIteration:
            self._close(frame)
        except Exception as exc:  # noqa: BLE001
            # a property that raises, a broken ``__iter__``
            frame.children = iter(())
            self._lines.append(f"{INDENT * frame.depth}<summary failed: {type(exc).__name__}>")
        else:
            frame.shown += 1
            self._visit(label, child, frame.depth)

    def _close(self, frame: _Frame) -> None:
        self._stack.pop()
        self._ancestors.discard(id(frame.data))
        if frame.total > frame.shown:
            self._lines.append(f"{INDENT * frame.depth}... (+{frame.total - frame.shown:,} more)")

    def _visit(self, label: Any, obj: Any, depth: int) -> None:
        r"""Write the line of an object, and open its frame if it is a
        container that is expanded."""
        max_characters = self._config.max_characters
        prefix = INDENT * depth
        if label is None:
            head = prefix
        else:
            text = label if type(label) is str else safe_str(label)
            head = f"{prefix}({_cut(text, max_characters)}): "
        summarizer = self._registry.find_summarizer(type(obj))
        try:
            text = self._describe(summarizer, obj, depth)
        except Exception as exc:  # noqa: BLE001
            # the data is broken, not the summary
            text = f"{type(obj)} <summary failed: {type(exc).__name__}>"
        line = head + text
        if depth > 0 and "\n" in line:
            # a text with several lines is aligned under its first character
            line = line.replace("\n", "\n" + prefix + INDENT)
        self._lines.append(line)

    def _describe(self, summarizer: Any, obj: Any, depth: int) -> str:
        r"""Return the text of an object, and open its frame if it is a
        container that is expanded."""
        config = self._config
        compact = depth >= config.max_depth
        result = summarizer.summarize(obj, config, compact=compact)
        if isinstance(result, str):
            return _cut(result, config.max_characters)
        if not isinstance(result, Node):
            msg = f"a summarizer must return a str or a Node, but got {type(result)}"
            raise TypeError(msg)
        if compact:
            return CompactRepr(config).repr(obj)
        if id(obj) in self._ancestors:
            return "<cycle>"
        children = iter(islice(result.children, config.max_items))
        self._stack.append(_Frame(obj, children, depth + 1, result.total))
        self._ancestors.add(id(obj))
        return result.header


def render(data: object, registry: SummarizerRegistry, config: SummaryConfig) -> str:
    r"""Render the summary of the data.

    Args:
        data: The data to summarize.
        registry: The registry that gives the summarizer of each type.
        config: The options of the summary.

    Returns:
        The summary.

    Raises:
        KeyError: If the registry has no summarizer for a type of the
            data, and not even for ``object``.

    Example:
        ```pycon
        >>> from coola.summary import SummaryConfig, get_default_registry
        >>> from coola.summary.engine import render
        >>> print(render([1, [2, 3]], get_default_registry(), SummaryConfig()))
        <class 'list'> (length=2)
          (0): 1
          (1): [2, 3]

        ```
    """
    return _Renderer(registry, config).render(data)
