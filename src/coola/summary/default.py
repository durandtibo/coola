r"""Implement a default summarizer for scalar and unknown types."""

from __future__ import annotations

__all__ = ["DefaultSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.base import BaseSummarizer
from coola.utils.safe import safe_str

if TYPE_CHECKING:
    from coola.summary.config import SummaryConfig


class DefaultSummarizer(BaseSummarizer[object]):
    r"""Implement a default summarizer that returns the ``str`` of the
    data.

    The summarizer is a leaf: it shows the type and the ``str`` of the
    data, and only the ``str`` when the data is at the display depth.
    The text is cut by the engine at ``max_characters``. It is
    registered for ``object``, so it is used for the types that have no
    other summarizer (numbers, strings, iterators, ...).

    Example:
        ```pycon
        >>> from coola.summary import DefaultSummarizer, SummaryConfig
        >>> summarizer = DefaultSummarizer()
        >>> print(summarizer.summarize(1, SummaryConfig(), compact=False))
        <class 'int'> 1
        >>> print(summarizer.summarize(1, SummaryConfig(), compact=True))
        1

        ```
    """

    def summarize(self, data: object, config: SummaryConfig, *, compact: bool) -> str:  # noqa: ARG002
        text = safe_str(data)
        if compact:
            return text
        return f"{type(data)} {text}"
