r"""Implement the default summarizer."""

from __future__ import annotations

__all__ = ["DefaultSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.base import BaseSummarizer
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry


class DefaultSummarizer(BaseSummarizer[object]):
    r"""Implement the default summarizer.

    Args:
        max_characters: The maximum number of characters to
            show. If a negative value is provided, all the characters
            are shown. Defaults to ``1000``, to keep the summary of an
            object with a huge ``str`` short. Note that ``str(data)``
            is computed before it is truncated.

    Example:
        ```pycon
        >>> from coola.summary import SummarizerRegistry, DefaultSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = DefaultSummarizer()
        >>> print(summarizer.summarize(1, registry))
        <class 'int'> 1

        ```
    """

    def __init__(self, max_characters: int = 1000) -> None:
        self._max_characters = max_characters

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(max_characters={self._max_characters:,})"

    def equal(self, other: object) -> bool:
        if not is_same_type(other, self):
            return False
        return self._max_characters == other._max_characters

    def summarize(
        self,
        data: object,
        registry: SummarizerRegistry,  # noqa: ARG002
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        text = self._summarize(_safe_str(data))
        if depth >= max_depth:
            return text
        return f"{type(data)} {text}"

    def _summarize(self, value: str) -> str:
        if self._max_characters >= 0 and len(value) > self._max_characters:
            value = value[: self._max_characters] + "..."
        return value


def _safe_str(data: object) -> str:
    r"""Convert the data to a string without ever raising.

    Falls back to a placeholder when ``str`` fails, e.g. a ``__str__``
    that raises or an ``int`` above the string conversion limit.
    """
    try:
        return str(data)
    except Exception:  # noqa: BLE001
        if isinstance(data, int):
            return f"<{type(data).__qualname__} with {data.bit_length():,} bits>"
        return f"<{type(data).__qualname__} object at {id(data):#x}>"
