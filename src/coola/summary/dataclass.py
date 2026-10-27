r"""Implement a summarizer for dataclasses."""

from __future__ import annotations

__all__ = ["DataclassSummarizer"]

import dataclasses
from typing import Any

from coola.summary.object_fields import BaseObjectFieldsSummarizer


class DataclassSummarizer(BaseObjectFieldsSummarizer[Any]):
    r"""Implement a summarizer for dataclasses.

    The children are the fields of the dataclass, labeled by their name.
    The fields declared with ``field(repr=False)`` are not shown, as they
    are not in the ``repr``: it is the standard way to hide a token or a
    large buffer.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.summary import DataclassSummarizer, SummaryConfig
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...     y: float
        >>> node = DataclassSummarizer().summarize(Point(1.0, 2.0), SummaryConfig(), compact=False)
        >>> list(node.children)
        [('x', 1.0), ('y', 2.0)]
        >>> node.total
        2
        >>> from dataclasses import field
        >>> @dataclass
        ... class User:
        ...     name: str
        ...     token: str = field(default="secret", repr=False)
        >>> node = DataclassSummarizer().summarize(User("a"), SummaryConfig(), compact=False)
        >>> list(node.children)
        [('name', 'a')]

        ```
    """

    def _get_names(self, data: Any) -> list[str]:
        return [field.name for field in dataclasses.fields(data) if field.repr]
