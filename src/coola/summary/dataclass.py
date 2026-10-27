r"""Implement a summarizer for dataclasses."""

from __future__ import annotations

__all__ = ["DataclassSummarizer"]

import dataclasses
from typing import Any

from coola.summary.object_fields import BaseObjectFieldsSummarizer


class DataclassSummarizer(BaseObjectFieldsSummarizer[Any]):
    r"""Implement a summarizer for dataclasses.

    The children are the fields of the dataclass, labeled by their name.

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

        ```
    """

    def _get_names(self, data: Any) -> list[str]:
        return [field.name for field in dataclasses.fields(data)]
