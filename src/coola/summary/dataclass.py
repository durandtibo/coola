r"""Implement a summarizer for dataclass instances."""

from __future__ import annotations

__all__ = ["DataclassSummarizer"]

import dataclasses
from typing import Any

from coola.summary.object_fields import BaseObjectFieldsSummarizer


class DataclassSummarizer(BaseObjectFieldsSummarizer[Any]):
    r"""Summarizer for dataclass instances.

    The fields are summarized with the registry (see
    ``BaseObjectFieldsSummarizer``), so ``max_depth`` and the
    summarizers of the field types apply. This summarizer is used for
    the dataclasses that have no summarizer of their own.

    Args:
        max_items: The maximum number of fields to display. If
            negative, all the fields are shown. Defaults to 5.
        num_spaces: The number of spaces for indenting each level.
            Defaults to 2.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.summary import DataclassSummarizer, get_default_registry
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...     y: float
        ...
        >>> print(DataclassSummarizer().summarize(Point(1.0, 2.0), get_default_registry()))
        <class 'coola.summary.dataclass.Point'> (fields=2)
          (x): 1.0
          (y): 2.0

        ```
    """

    def _get_fields(self, data: Any) -> dict[str, Any]:
        return {field.name: getattr(data, field.name) for field in dataclasses.fields(data)}
