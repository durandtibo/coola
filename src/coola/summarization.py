r"""Implement the main summarization public features."""

from __future__ import annotations

__all__ = ["summary"]


from typing import TYPE_CHECKING, Any

from coola.summarizers.summarizer import Summarizer

if TYPE_CHECKING:
    from coola.summarizers.base import BaseSummarizer


def summary(value: Any, max_depth: int = 1, summarizer: BaseSummarizer | None = None) -> str:
    r"""Summarize the input value in a string.

    Args:
        value: The value to summarize.
        max_depth: The maximum depth to summarize if the
            input is nested.
        summarizer: The summarization strategy. If ``None``,
            the default ``Summarizer`` is used.

    Returns:
        The summary as a string.

    Example usage:

    ```pycon

    >>> from coola import summary
    >>> print(summary(1))
    <class 'int'> 1
    >>> print(summary(["abc", "def"]))
    <class 'list'> (length=2)
      (0): abc
      (1): def
    >>> print(summary([[0, 1, 2], {"key1": "abc", "key2": "def"}]))
    <class 'list'> (length=2)
      (0): [0, 1, 2]
      (1): {'key1': 'abc', 'key2': 'def'}
    >>> print(summary([[0, 1, 2], {"key1": "abc", "key2": "def"}], max_depth=2))
    <class 'list'> (length=2)
      (0): <class 'list'> (length=3)
          (0): 0
          (1): 1
          (2): 2
      (1): <class 'dict'> (length=2)
          (key1): abc
          (key2): def

    ```
    """
    summarizer = summarizer or Summarizer()
    return summarizer.summary(value=value, depth=0, max_depth=max_depth)
